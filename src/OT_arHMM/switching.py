import lumicks.pylake as lk
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import welch
from scipy.optimize import curve_fit
from statsmodels.tsa.api import MarkovRegression, MarkovAutoregression


class Switching():
    def __init__(self, file, downsample=4, load=False):
        if not load:
            self.file = str(file)
            self.downsample = downsample
            pylake_file = lk.File(self.file)

            self.raw_time = pylake_file.force1x.seconds
            self.raw_force = (pylake_file.force2x.data-pylake_file.force1x.data)/2
            self.raw_f_s = pylake_file.force1x.sample_rate
            self.raw_N = len(self.raw_time)

            self.time = self.raw_time[::downsample]
            self.force = self.raw_force[::downsample]
            self.f_s = self.raw_f_s/downsample
            self.N = len(self.time)

            self.gmm = {}
            self.psd = {}
            self.raw_psd = {}
            self.psd_fit = {}
            self.full_psd_fit = {}
            self.hmm = {}
            self.arhmm = {}
            self.hmm_lts = {}
            self.arhmm_lts = {}

        else:
            with h5py.File(file, "r") as f:
                for key, value in f.attrs.items():
                    setattr(self, key, value)

                self.raw_time = f["raw_time"][:]
                self.raw_force = f["raw_force"][:]

                self.time = f["time"][:]
                self.force = f["force"][:]
                
                self.gmm = {}
                if "gmm" in f:
                    self.gmm = dict(f["gmm"].attrs)
                
                self.psd = {}
                if "psd" in f:
                    self.psd = {
                        "f": f["psd"]["f"][:],
                        "value": f["psd"]["value"][:]
                    }
    
                self.raw_psd = {}
                if "raw_psd" in f:
                    self.raw_psd = {
                        "f": f["raw_psd"]["f"][:],
                        "value": f["raw_psd"]["value"][:]
                    }

                self.psd_fit = {}
                if "psd_fit" in f:
                    self.psd_fit = dict(f["psd_fit"].attrs)
                
                self.full_psd_fit = {}
                if "full_psd_fit" in f:
                    self.full_psd_fit = dict(f["full_psd_fit"].attrs)
                
                self.hmm = {}
                if "hmm" in f:
                    if "initial params" in f["hmm"]:
                        self.hmm["initial params"] = dict(f["hmm"]["initial params"].attrs)
                    if "params" in f["hmm"]:
                        self.hmm["params"] = dict(f["hmm"]["params"].attrs)
                    if "fit" in f["hmm"]:
                        self.hmm["fit"] = dict(f["hmm"]["fit"].attrs)
                    if "labels" in f["hmm"]:
                        self.hmm["labels"] = f["hmm"]["labels"][:]

                self.arhmm = {}
                if "arhmm" in f:
                    if "initial params" in f["arhmm"]:
                        self.arhmm["initial params"] = dict(f["arhmm"]["initial params"].attrs)
                    if "params" in f["arhmm"]:
                        self.arhmm["params"] = dict(f["arhmm"]["params"].attrs)
                    if "fit" in f["arhmm"]:
                        self.arhmm["fit"] = dict(f["arhmm"]["fit"].attrs)
                    if "labels" in f["arhmm"]:
                        self.arhmm["labels"] = f["arhmm"]["labels"][:]

                self.hmm_lts = {}
                if "hmm_lts" in f:
                    if "transitions" in f["hmm_lts"]:
                        self.hmm_lts["transitions"] = f["hmm_lts"]["transitions"][:]
                    if "states" in f["hmm_lts"]:
                        self.hmm_lts["states"] = f["hmm_lts"]["states"][:]
                    if "lengths" in f["hmm_lts"]:
                        self.hmm_lts["lengths"] = f["hmm_lts"]["lengths"][:]
                    if "unfolded lts" in f["hmm_lts"]:
                        self.hmm_lts["unfolded lts"] = f["hmm_lts"]["unfolded lts"][:]
                    if "folded lts" in f["hmm_lts"]:
                        self.hmm_lts["folded lts"] = f["hmm_lts"]["folded lts"][:]
                    if "rates" in f["hmm_lts"].attrs:
                        self.hmm_lts["rates"] = f["hmm_lts"].attrs["rates"]

                self.arhmm_lts = {}
                if "arhmm_lts" in f:
                    if "transitions" in f["arhmm_lts"]:
                        self.arhmm_lts["transitions"] = f["arhmm_lts"]["transitions"][:]
                    if "states" in f["arhmm_lts"]:
                        self.arhmm_lts["states"] = f["arhmm_lts"]["states"][:]
                    if "lengths" in f["arhmm_lts"]:
                        self.arhmm_lts["lengths"] = f["arhmm_lts"]["lengths"][:]
                    if "unfolded lts" in f["arhmm_lts"]:
                        self.arhmm_lts["unfolded lts"] = f["arhmm_lts"]["unfolded lts"][:]
                    if "folded lts" in f["arhmm_lts"]:
                        self.arhmm_lts["folded lts"] = f["arhmm_lts"]["folded lts"][:]
                    if "rates" in f["arhmm_lts"].attrs:
                        self.arhmm_lts["rates"] = f["arhmm_lts"].attrs["rates"]

    def __str__(self):
        out = ""
        out += f"Switching object for file {self.file}\n"
        out += "\n"
        
        out += "Raw data parameters:\n"
        out += f"\tLength: {self.raw_N} elements\n"
        out += f"\tSample rate: {self.raw_f_s/1000:.3f} kHz\n"
        out += f"\tDuration: {self.raw_N/self.raw_f_s:.2f}s\n"
        out += "\n"
        
        out += "Downsampled data parameters:\n"
        out += f"\tDownsampling factor: {self.downsample}\n"
        out += f"\tLength: {self.N} elements\n"
        out += f"\tSample rate: {self.f_s/1000:.3f} kHz\n"
        out += "\n"

        if {"probabilities", "means", "stds"} <= self.gmm.keys():
            out += "GMM has been fit with:\n"
            out += f"\tProbabilities = {self.gmm["probabilities"][0]*100:.1f}%, {self.gmm["probabilities"][1]*100:.1f}\n"
            out += f"\tMeans = {self.gmm["means"][0]:.1f} pN, {self.gmm["means"][1]:.1f} pN\n"
            out += f"\tStd dev = {self.gmm["stds"][0]:.2f} pN, {self.gmm["stds"][1]:.2f} pN\n"
        else:
            out += f"GMM has not been fit\n"
        out += "\n"

        if {"value", "f"} <= self.psd.keys():
            out += "Downsampled PSD has been calculated\n"
        else:
            out += "Downsampled PSD has not been calculated\n"
        if {"value", "f"} <= self.raw_psd.keys():
            out += "Raw PSD has been calculated\n"
        else:
            out += "Raw PSD has not been calculated\n"
        out += "\n"

        if {"A", "f_c"} <= self.psd_fit.keys():
            out += "Switching Lorentzian has been fit with:\n"
            out += f"\tAmplitude: {self.psd_fit["A"]:.2g} pN^2/Hz\n"
            out += f"\tCorner frequency {self.psd_fit["f_c"]:.1f} Hz\n"
        else:
            out += f"Switching Lorentzian has not been fit\n"
        out += "\n"

        if {"A_1", "f_c_1", "A_2", "f_c_2"} <= self.full_psd_fit.keys():
            out += f"Double Lorentzian has been fit with:\n"
            out += f"\tAmplitude 1 = {self.full_psd_fit["A_1"]:.2g} pN^2/Hz\n"
            out += f"\tCorner frequency 1 = {self.full_psd_fit["f_c_1"]:.1f}  Hz\n"
            out += f"\tAmplitude 2 = {self.full_psd_fit["A_2"]:.2g} pN^2/Hz\n"
            out += f"\tCorner frequency 2 = {self.full_psd_fit["f_c_2"]:.0f} Hz\n"
        else:
            out += f"Double Lorentzian has not been fit\n"
        out += "\n"

        if {"params", "fit"} <= self.hmm.keys():
            out += "HMM has been fit with:\n"
            out += f"\tMeans: {self.hmm["params"]["means"][0]:.1f} pN, {self.hmm["params"]["means"][1]:.1f} pN\n"
            out += f"\tRates: {self.hmm["params"]["rates"][0]:.1f} Hz, {self.hmm["params"]["rates"][1]:.1f} Hz\n"
            out += f"\tStd dev: {np.sqrt(self.hmm["params"]["variance"]):.2f} pN\n"
            out += f"\tBIC: {(self.hmm["fit"]["bic"]):.2g}\n"
            out += f"\tLog likelihood: {(self.hmm["fit"]["loglikelihood"]):.2g}\n"
        else:
            out += "HMM has not been fit\n"
        out += "\n"
        
        if {"params", "fit"} <= self.arhmm.keys():
            out += "ARHMM has been fit with:\n"
            out += f"\tMeans: {self.arhmm["params"]["means"][0]:.1f} pN, {self.arhmm["params"]["means"][1]:.1f} pN\n"
            out += f"\tRates: {self.arhmm["params"]["rates"][0]:.1f} Hz, {self.arhmm["params"]["rates"][1]:.1f} Hz\n"
            out += f"\tAutocorrelations: {self.arhmm["params"]["alphas"][0]:.2f}, {self.arhmm["params"]["alphas"][1]:.2f}\n"
            out += f"\tStd dev: {np.sqrt(self.arhmm["params"]["variance"]):.2f} pN\n"
            out += f"\tBIC: {(self.arhmm["fit"]["bic"]):.2g}\n"
            out += f"\tLog likelihood: {(self.arhmm["fit"]["loglikelihood"]):.2g}\n"
        else:
            out += "ARHMM has not been fit\n"

        return out


    def save(self, file):
        with h5py.File(file, "w") as f:
            attrs = ["file", "downsample", "f_s", "raw_f_s", "N", "raw_N"]
            for a in attrs:
                f.attrs[a] = getattr(self, a)

            datasets = ["time", "force", "raw_time", "raw_force"]
            for d in datasets:
                f.create_dataset(d, data=getattr(self, d), compression="gzip")

            if self.gmm:
                gmm = f.create_group("gmm")
                for key, value in self.gmm.items():
                    gmm.attrs[key] = value
            
            if self.psd:
                psd = f.create_group("psd")
                for key, value in self.psd.items():
                    psd.create_dataset(key, data=value, compression="gzip")
            
            if self.raw_psd:
                raw_psd = f.create_group("raw_psd")
                for key, value in self.raw_psd.items():
                    raw_psd.create_dataset(key, data=value, compression="gzip")

            if self.psd_fit:
                psd_fit = f.create_group("psd_fit")
                for key, value in self.psd_fit.items():
                    psd_fit.attrs[key] = value

            if self.full_psd_fit:
                full_psd_fit = f.create_group("full_psd_fit")
                for key, value in self.full_psd_fit.items():
                    full_psd_fit.attrs[key] = value

            if self.hmm:
                hmm = f.create_group("hmm")

                hmm_init = hmm.create_group("initial params")
                for key, value in self.hmm["initial params"].items():
                    hmm_init.attrs[key] = value
                
                hmm_params = hmm.create_group("params")
                for key, value in self.hmm["params"].items():
                    hmm_params.attrs[key] = value

                hmm_fit = hmm.create_group("fit")
                for key, value in self.hmm["fit"].items():
                    hmm_fit.attrs[key] = value

                hmm.create_dataset("labels", data=self.hmm["labels"], compression="gzip")
            
            if self.arhmm:
                arhmm = f.create_group("arhmm")

                arhmm_init = arhmm.create_group("initial params")
                for key, value in self.arhmm["initial params"].items():
                    arhmm_init.attrs[key] = value
                
                arhmm_params = arhmm.create_group("params")
                for key, value in self.arhmm["params"].items():
                    arhmm_params.attrs[key] = value

                arhmm_fit = arhmm.create_group("fit")
                for key, value in self.arhmm["fit"].items():
                    arhmm_fit.attrs[key] = value

                arhmm.create_dataset("labels", data=self.arhmm["labels"], compression="gzip")
        
            if self.hmm_lts:
                hmm_lts = f.create_group("hmm_lts")

                for dset in ["transitions", "states", "lengths", "unfolded lts", "folded lts"]:
                    hmm_lts.create_dataset(dset, data=self.hmm_lts[dset], compression="gzip")
                hmm_lts.attrs["rates"] = self.hmm_lts["rates"]

            if self.arhmm_lts:
                arhmm_lts = f.create_group("arhmm_lts")

                for dset in ["transitions", "states", "lengths", "unfolded lts", "folded lts"]:
                    arhmm_lts.create_dataset(dset, data=self.arhmm_lts[dset], compression="gzip")
                arhmm_lts.attrs["rates"] = self.arhmm_lts["rates"]


    def plot_force_time(self, labels=None, raw=False, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if labels is not None:
            kwargs["c"] = ["tab:orange" if x else "tab:blue" for x in labels]

        if "s" not in kwargs:
            kwargs["s"] = 0.1

        ax.scatter(self.time[1:], self.force[1:], **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (pN)")

        return ax
    
    
    def plot_force_hist(self, binwidth=0.05, raw=False, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        bins = np.arange(min(self.force), max(self.force) + binwidth, binwidth)
        ax.hist(self.force, bins=bins, density=True, **kwargs)
        ax.set_xlabel("Force (pN)")
        ax.set_ylabel("Probability density (pN$^{-1}$)")

        return ax
    

    def fit_gmm(self, probs_guess, means_guess, stds_guess):
        self.gmm = {}
        
        means_init = np.reshape(means_guess, (2,1))
        precisions_init = np.array(stds_guess)**-2
        force = self.force.reshape((-1, 1))
        
        gmm = GaussianMixture(2,
            covariance_type="spherical",
            weights_init=probs_guess,
            means_init=means_init,
            precisions_init=precisions_init)
        fit = gmm.fit(force)
        
        probabilities = fit.weights_
        means = fit.means_.flatten()
        variances = fit.covariances_
        
        if means[0]>means[1]:
            probabilities = np.flip(probabilities)
            means = np.flip(means)
            variances = np.flip(variances)

        self.gmm["probabilities"] = probabilities
        self.gmm["means"] = means
        self.gmm["variances"] = variances
        self.gmm["stds"] = np.sqrt(variances)


    def plot_gmm_pdf(self, annotate=True, ax=None):
        gmm = self.gmm
        means = gmm["means"]
        stds = gmm["stds"]
        probs = gmm["probabilities"]


        forces = np.linspace(min(self.force), max(self.force), 100)
        forces = forces.reshape((-1, 1))

        pdfs = [probs[i]*norm(loc=means[i], scale=stds[i]).pdf(forces) for i in [0,1]]
        sum_pdf = pdfs[0]+pdfs[1]
        
        if ax is None:
            ax = plt.gca()
        ax.plot(forces, pdfs[0], color="tab:blue", label="Peak 1")
        ax.plot(forces, pdfs[1], color="tab:orange", label="Peak 2")
        ax.plot(forces, sum_pdf, color="black", label="Total")

        if annotate:
            txt = ""
            for i in [0,1]:
                txt += f"$p_{i+1}={probs[i]:.2f}$\n"
                txt += f"$\\mu_{i+1}={means[i]:.2f}$ pN\n"
                txt += f"$\\sigma_{i+1}={stds[i]:.2f}$ pN\n"
                if i==0:
                    txt += "\n"
            ax.text(.05, .95, txt, va="top", transform=ax.transAxes)

        ax.set_xlabel("Force (pN)")
        ax.set_ylabel("Probability density (pN$^{-1}$)")
        return ax
    

    def calculate_psd(self, n_segments=10):
        f, psd = welch(self.force, self.f_s, nperseg=int(self.N/n_segments))
        raw_f, raw_psd = welch(self.raw_force, self.raw_f_s, nperseg=int(self.raw_N/n_segments))
        self.psd = {"value": psd, "f": f}
        self.raw_psd = {"value": raw_psd, "f": raw_f}

    
    def plot_psd(self, raw=False, log="loglog", ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        if raw:
            data = self.raw_psd
        else:
            data = self.psd
        ax.plot(data["f"], data["value"], **kwargs)

        if log=="loglog":
            ax.loglog()
        elif log=="semilogx":
            ax.semilogx()
        elif log=="semilogy":
            ax.semilogy()
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (pN$^2$/Hz)")

        return ax
    

    @staticmethod
    def _lorentzian(f, A, f_c):
        return A/(1+(f/f_c)**2)

    
    @staticmethod
    def _lorentzian_fit(f, A, f_c2):
        return A/(1+f**2/f_c2)
    

    @staticmethod
    def _double_lorentzian(f, A_1, f_c_1, A_2, f_c_2):
        return A_1/(1+(f/f_c_1)**2) + A_2/(1+(f/f_c_2)**2)
   

    @staticmethod
    def _double_lorentzian_fit(f, A_1, f_c2_1, A_2, f_c2_2):
        return A_1/(1+f**2/f_c2_1) + A_2/(1+f**2/f_c2_2)


    def fit_psd(self, A_guess, f_c_guess, f_min=0, f_max=30, raw=False):
        if raw:
            psd = self.raw_psd
        else:
            psd = self.psd

        f = psd["f"]
        value = psd["value"]

        mask = (f>f_min) & (f<f_max)
        f_mask = f[mask]
        value_mask = value[mask]

        guess = (A_guess, f_c_guess**2)

        fit = curve_fit(self._lorentzian_fit, f_mask, value_mask, p0=guess, sigma=np.sqrt(value_mask))

        self.psd_fit = {"A": fit[0][0], "f_c": np.sqrt(fit[0][1])}


    def fit_full_psd(self, A_1_guess, f_c_1_guess, A_2_guess, f_c_2_guess, f_min=0, f_max=2e4, raw=True):
        if raw:
            psd = self.raw_psd
        else:
            psd = self.psd

        f = psd["f"]
        value = psd["value"]

        mask = (f>f_min) & (f<f_max)
        f_mask = f[mask]
        value_mask = value[mask]

        guess = (A_1_guess, f_c_1_guess**2, A_2_guess, f_c_2_guess**2)

        fit = curve_fit(self._double_lorentzian_fit, f_mask, value_mask, p0=guess, sigma=np.sqrt(value_mask))

        self.full_psd_fit = {"A_1": fit[0][0], "f_c_1": np.sqrt(fit[0][1]), "A_2": fit[0][2], "f_c_2": np.sqrt(fit[0][3])}


    def plot_psd_fit(self, log="loglog", ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        f = self.raw_psd["f"]

        ax.plot(f, self._lorentzian(f, **self.psd_fit), **kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (pN$^2$/Hz)")

        if log in ("loglog", "semilogx"):
            ax.set_xscale("log")

        if log in ("loglog", "semilogy"):
            ax.set_yscale("log")

        return ax


    def plot_full_psd_fit(self, log="loglog", plot_all = True, ax=None, subplot1_kws={}, subplot2_kws={}, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        f = self.raw_psd["f"]

        if plot_all:
            ax.plot(f, self._lorentzian(f, self.full_psd_fit["A_1"], self.full_psd_fit["f_c_1"]), **subplot1_kws)
            ax.plot(f, self._lorentzian(f, self.full_psd_fit["A_2"], self.full_psd_fit["f_c_2"]), **subplot2_kws)
        ax.plot(f, self._double_lorentzian(f, **self.full_psd_fit), **kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (pN$^2$/Hz)")

        if log in ("loglog", "semilogx"):
            ax.set_xscale("log")

        if log in ("loglog", "semilogy"):
            ax.set_yscale("log")

        return ax
    

    @staticmethod
    def _hmm_sm_params_to_dict(sm_params, f_s):
        p_11, p_21, *means, variance = sm_params
        k_1 = (1-p_11)*f_s
        k_2 = p_21*f_s
        rates = np.array([k_1, k_2])
        omega_c = k_1+k_2
        probabilities = np.flip(rates)/omega_c
        f_c = omega_c/2/np.pi

        dict_params = {
            "f_c": f_c,
            "omega_c": omega_c,
            "rates": rates,
            "probabilities": probabilities,
            "transition matrix": np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            "means": means,
            "variance": variance,
            "params": sm_params
        }

        return dict_params
    

    def _hmm_get_default_params(self):
        f_s = self.f_s
        f_c = self.psd_fit["f_c"]    
        means = self.gmm["means"]
        probabilities = self.gmm["probabilities"]
        omega_c = 2*np.pi*f_c
        k_2, k_1 = probabilities*omega_c
        rates = np.array([k_1, k_2])
        p_11 = 1-k_1/f_s
        p_21 = k_2/f_s

        variance = np.mean(self.gmm["variances"])
        sm_params = np.array([p_11, p_21, *means, variance])

        dict_params = {
            "f_c": f_c,
            "omega_c": omega_c,
            "rates": rates,
            "probabilities": probabilities,
            "transition matrix": np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            "means": means,
            "variance": variance,
            "params": sm_params
        }

        return dict_params


    def fit_hmm(self, params=None):
        force = self.force

        if params is None:
            init_params = self._hmm_get_default_params()
        else:
            init_params = self._hmm_sm_params_to_dict(params, self.f_s)

        hmm = MarkovRegression(force, 2)
        hmm_fit = hmm.fit(init_params["params"])

        out_params = self._hmm_sm_params_to_dict(hmm_fit.params, self.f_s)

        labels = np.argmax(hmm_fit.smoothed_marginal_probabilities, axis=1)[1:]

        fit_params = {
            "standard error": hmm_fit.bse,
            "loglikelihood": hmm_fit.llf,
            "bic": hmm_fit.bic,
            "aic": hmm_fit.aic,
            "hqic": hmm_fit.hqic
        }

        self.hmm = {"initial params": init_params, "params": out_params, "labels": labels, "fit": fit_params}

    @staticmethod
    def _arhmm_sm_params_to_dict(sm_params, f_s):
        p_11, p_21, *means, variance, alpha_1, alpha_2 = sm_params
        alphas = np.array([alpha_1, alpha_2])

        k_1 = (1-p_11)*f_s
        k_2 = p_21*f_s

        rates = np.array([k_1, k_2])
        omega_c_1 = k_1+k_2
        probabilities = np.flip(rates)/omega_c_1
        f_c_1 = omega_c_1/2/np.pi
        omega_c_2 = -f_s*np.mean(np.log([alpha_1, alpha_2]))
        f_c_2 = omega_c_2/2/np.pi

        dict_params = {
            "f_c_1": f_c_1,
            "f_c_2": f_c_2,
            "omega_c_1": omega_c_1,
            "omega_c_2": omega_c_2,
            "rates": rates,
            "probabilities": probabilities,
            "transition matrix": np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            "means": means,
            "alphas": alphas,
            "variance": variance,
            "params": sm_params
        }

        return dict_params
    

    def _arhmm_get_default_params(self):
        f_s = self.f_s
        f_c_1 = self.full_psd_fit["f_c_1"]
        f_c_2 = self.full_psd_fit["f_c_2"]
        means = self.gmm["means"]
        probabilities = self.gmm["probabilities"]

        omega_c_1 = 2*np.pi*f_c_1
        k_2, k_1 = probabilities*omega_c_1
        rates = np.array([k_1, k_2])
        p_11 = 1-k_1/f_s
        p_21 = k_2/f_s

        omega_c_2 = 2*np.pi*f_c_2
        alpha_1 = np.exp(-omega_c_2/f_s)
        alpha_2 = alpha_1
        alphas = np.array([alpha_1, alpha_2])

        variance = np.mean(self.gmm["variances"])*(1-alpha_1**2)

        sm_params = np.array([p_11, p_21, *means, variance, alpha_1, alpha_2])

        dict_params = {
            "f_c_1": f_c_1,
            "f_c_2": f_c_2,
            "omega_c_1": omega_c_1,
            "omega_c_2": omega_c_2,
            "rates": rates,
            "probabilities": probabilities,
            "transition matrix": np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            "means": means,
            "alphas": alphas,
            "variance": variance,
            "params": sm_params
        }

        return dict_params


    def fit_arhmm(self, params=None):
        force = self.force

        if params is None:
            init_params = self._arhmm_get_default_params()
        else:
            init_params = self._arhmm_sm_params_to_dict(params, self.f_s)

        arhmm = MarkovAutoregression(force, 2, 1)
        arhmm_fit = arhmm.fit(init_params["params"])

        out_params = self._arhmm_sm_params_to_dict(arhmm_fit.params, self.f_s)

        labels = np.argmax(arhmm_fit.smoothed_marginal_probabilities, axis=1)

        fit_params = {
            "standard error": arhmm_fit.bse,
            "loglikelihood": arhmm_fit.llf,
            "bic": arhmm_fit.bic,
            "aic": arhmm_fit.aic,
            "hqic": arhmm_fit.hqic
        }

        self.arhmm = {"initial params": init_params, "params": out_params, "labels": labels, "fit": fit_params}

    
    @staticmethod
    def _calculate_lifetimes(labels, f_s):
        # transitions: last index before a state transition
        # states: label of state before transition
        # lengths: how long does state last until it transitions
        # Folding is 0->1
        # Unfolding is 1->0

        delta = np.diff(labels)
        transitions = np.nonzero(delta)[0]

        states = labels[transitions]

        lengths = np.diff(np.insert(transitions, 0, -1))
        lifetimes = lengths/f_s

        unfolded_lts = lifetimes[states==0]
        folded_lts = lifetimes[states==1]

        folding_rate = 1/np.mean(unfolded_lts)
        unfolding_rate = 1/np.mean(folded_lts)

        out = {
            "transitions": transitions,
            "states": states,
            "lengths": lengths,
            "unfolded lts": unfolded_lts,
            "folded lts": folded_lts,
            "rates": (folding_rate, unfolding_rate)
        }

        return out
    
    
    def calculate_lifetimes(self, hmm=True, arhmm=True):
        if hmm:
            self.hmm_lts = self._calculate_lifetimes(self.hmm["labels"], self.f_s)
            
        if arhmm:
            self.arhmm_lts = self._calculate_lifetimes(self.arhmm["labels"], self.f_s)
        

    @staticmethod
    def _plot_lts(lifetimes, ax, fill, log, **kwargs):
        if ax is None:
            ax = plt.gca()

        lts_sorted = sorted(lifetimes)

        survival = np.flip(np.linspace(0, 1, num=len(lifetimes)))
        
        if fill:
            ax.fill_between(lts_sorted, survival, step="post", **kwargs)
        else:
            ax.plot(lts_sorted, survival, drawstyle="steps-post", **kwargs)
        
        if log in ["loglog", "semilogx"]:
            ax.set_xscale("log")
        if log in ["loglog", "semilogy"]:
            ax.set_yscale("log")

        ax.set_xlabel("Time (s)")

        return ax
    

    def plot_folding(self, ax=None, fill=False, log=None, hmm=False, **kwargs):
        if hmm:
            lifetimes = self.hmm_lts["unfolded lts"]
        else:
            lifetimes = self.arhmm_lts["unfolded lts"]

        ax = self._plot_lts(lifetimes, ax, fill, log, **kwargs)

        ax.set_ylabel("Fraction unfolded")

        return ax
    

    def plot_unfolding(self, ax=None, fill=False, log=None, hmm=False, **kwargs):
        if hmm:
            lifetimes = self.hmm_lts["folded lts"]
        else:
            lifetimes = self.arhmm_lts["folded lts"]

        ax = self._plot_lts(lifetimes, ax, fill, log, **kwargs)

        ax.set_ylabel("Fraction folded")

        return ax
    

    @staticmethod
    def _plot_rate(rate, time_lims, ax, log, **kwargs):
        if ax is None:
            ax = plt.gca()

        times = np.linspace(time_lims[0], time_lims[1], 200)

        ax.plot(times, np.exp(-rate*times), **kwargs)

        if log in ["loglog", "semilogx"]:
            ax.set_xscale("log")
        if log in ["loglog", "semilogy"]:
            ax.set_yscale("log")

        ax.set_xlabel("Time (s)")

        return ax


    def plot_folding_rate(self, from_lts=True, label_rate=True, min_time=0, max_time=None, ax=None, log=None, hmm=False, **kwargs):
        if hmm:
            model = self.hmm
            lts = self.hmm_lts
        else:
            model = self.arhmm
            lts = self.arhmm_lts

        if from_lts:
            rate = lts["rates"][0]
        else:
            rate = model["params"]["rates"][0]

        if max_time is None:
            max_time = np.max(lts["unfolded lts"])

        if label_rate:
            if from_lts:
                type_str = "lt"
            else:
                type_str = "mod"

            if hmm:
                model_str = "hmm"
            else:
                model_str = "arhmm"

            label = f"$k_{{\\mathrm{{{type_str},{model_str}}}}} = {rate:.1f}$ Hz"
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, label=label, **kwargs)
        else:
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, **kwargs)

        ax.set_ylabel("Fraction unfolded")
        
        return ax


    def plot_unfolding_rate(self, from_lts=True, label_rate=True, min_time=0, max_time=None, ax=None, log=None, hmm=False, **kwargs):
        if hmm:
            model = self.hmm
            lts = self.hmm_lts
        else:
            model = self.arhmm
            lts = self.arhmm_lts

        if from_lts:
            rate = lts["rates"][1]
        else:
            rate = model["params"]["rates"][1]

        if max_time is None:
            max_time = np.max(lts["folded lts"])

        if label_rate:
            if from_lts:
                type_str = "lt"
            else:
                type_str = "mod"

            if hmm:
                model_str = "hmm"
            else:
                model_str = "arhmm"

            label = f"$k_{{\\mathrm{{{type_str},{model_str}}}}} = {rate:.1f}$ Hz"
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, label=label, **kwargs)
        else:
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, **kwargs)

        ax.set_ylabel("Fraction folded")

        return ax
