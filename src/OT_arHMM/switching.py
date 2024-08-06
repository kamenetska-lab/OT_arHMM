import lumicks.pylake as lk
import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from scipy.signal import welch
from scipy.optimize import curve_fit
from statsmodels.tsa.api import MarkovRegression, MarkovAutoregression
from dataclasses import dataclass
from pathlib import Path
from functools import partial


def lorentzian(f: float, A: float, f_c: float):
    """A Lorentzian with amplitude A and corner frequency f_c."""
    return A/(1+(f/f_c)**2)


def double_lorentzian(f: float, A_1: float, f_c_1: float, A_2: float, f_c_2: float):
    """A double Lorentzian with amplitudes A_i and corner frequencies f_c_i."""
    return lorentzian(f, A_1, f_c_1) + lorentzian(f, A_2, f_c_2)


def aliased_lorentzian(f: float, A: float, f_c: float, f_s: float):
    """An aliased Lorentzian with amplitude A and corner frequency f_c."""
    x = np.exp(-2*np.pi*f_c/f_s)
    return A*(1-x**2)/(x**2-2*x*np.cos(2*np.pi*f/f_s)+1)*np.tanh(np.pi*f_c/f_s)


def aliased_double_lorentzian(f: float, A_1: float, f_c_1: float, A_2: float, f_c_2: float, f_s: float):
    """An aliased double Lorentzian with amplitudes A_i and corner frequency f_c_i."""
    return aliased_lorentzian(f, A_1, f_c_1, f_s) + aliased_lorentzian(f, A_2, f_c_2, f_s)


class h5Group:
    """Class to facilitate saving/loading to/from h5 files."""    
    def save(self, h5: str | Path | h5py.Group, create_file=True):
        if create_file:
            h5 = h5py.File(h5, 'w')

        try:
            for name in self._datasets:
                dset = getattr(self, name)
                # Compression throws an error if data is actually scalar
                if np.ndim(dset) == 0:
                    h5.create_dataset(name, data=dset)
                else:
                    h5.create_dataset(name, data=dset, compression="gzip")  
            
            for name in self._attrs:
                h5.attrs[name] = getattr(self, name)

            for name in self._subgrps:
                subgrp = getattr(self, name)
                if subgrp is not None:
                    h5_subgrp = h5.create_group(name)
                    subgrp.save(h5_subgrp, create_file=False)

        finally:
            if create_file:
                h5.close()


    @classmethod
    def reload(cls, h5: str | Path | h5py.Group, open_file=True):
        if open_file:
            h5 = h5py.File(h5, 'r') 
        
        try:
            params = {}

            for name in cls._datasets:
                params[name] = h5[name][()]
            
            for name in cls._attrs:
                params[name] = h5.attrs[name]

            for name in cls._subgrps:
                if name in h5:
                    params[name] = cls._subgrps[name].reload(h5[name], open_file=False)
                else:
                    params[name] = None
            
        finally:
            if open_file:
                h5.close()

        return cls(**params)
    

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class PSD_results(h5Group):
    f: np.ndarray
    value: np.ndarray
    
    _datasets = ["f", "value"]
    _attrs = []
    _subgrps = {}


@dataclass
class PSD_fit(h5Group):
    A: np.ndarray
    f_c: np.ndarray

    _datasets = ["A", "f_c"]
    _attrs = []
    _subgrps = {}


@dataclass
class GMM_results(h5Group):
    means: np.ndarray
    probabilities: np.ndarray
    stds: np.ndarray
    variances: np.ndarray

    _datasets = ["means", "probabilities", "stds", "variances"]
    _attrs = []
    _subgrps = {}

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class HMM_params(h5Group):
    means: np.ndarray
    variances: np.ndarray
    probabilities: np.ndarray
    f_cs: np.ndarray
    omega_cs: np.ndarray
    rates: np.ndarray
    alphas: np.ndarray
    transition_matrix: np.ndarray
    params: np.ndarray

    _datasets = ["means", "variances", "probabilities", "f_cs", "omega_cs", "rates", "alphas", "transition_matrix", "params"]
    _attrs = []
    _subgrps = {}
    

@dataclass
class Fit_quality(h5Group):
    standard_error: np.ndarray
    loglikelihood: float
    bic: float
    aic: float
    hqic: float

    _datasets = []
    _attrs = ["standard_error", "loglikelihood", "bic", "aic", "hqic"]
    _subgrps = {}


@dataclass
class LT_results(h5Group):
    transitions: np.ndarray
    states: np.ndarray
    lengths: np.ndarray
    unfolded_lts: np.ndarray
    folded_lts: np.ndarray
    rates: np.ndarray

    _datasets = ["transitions", "states", "lengths", "unfolded_lts", "folded_lts", "rates"]
    _attrs = []
    _subgrps = {}


@dataclass
class HMM_results(h5Group):
    initial_params: HMM_params
    fit_params: HMM_params
    fit_quality: Fit_quality
    labels: np.ndarray
    smoothed_probabilities: np.ndarray
    lifetimes: LT_results

    _datasets = ["labels", "smoothed_probabilities"]
    _attrs = []
    _subgrps = {"initial_params": HMM_params, "fit_params": HMM_params, "fit_quality": Fit_quality, "lifetimes": LT_results}


@dataclass
class Switching(h5Group):
    file: str

    raw_time: np.ndarray
    raw_force: np.ndarray
    raw_f_s: float
    raw_N: int
    
    downsample: int

    time: np.ndarray
    force: np.ndarray
    f_s: float
    N: int

    gmm: GMM_results | None = None
    psd: PSD_results | None = None
    raw_psd: PSD_results | None = None

    psd_fit: PSD_fit | None = None

    wnhmm: HMM_results | None = None
    arhmm: HMM_results | None = None

    _datasets = ["raw_time", "raw_force", "time", "force"]
    _attrs = ["file", "raw_f_s", "raw_N", "downsample", "f_s", "N"]
    _subgrps = {"gmm": GMM_results, "psd": PSD_results, "raw_psd": PSD_results, "psd_fit": PSD_fit, "wnhmm": HMM_results, "arhmm": HMM_results} 


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

        if self.gmm:
            out += "GMM has been fit with:\n"
            out += f"\tProbabilities = {self.gmm.probabilities[0]*100:.1f}%, {self.gmm.probabilities[1]*100:.1f}%\n"
            out += f"\tMeans = {self.gmm.means[0]:.1f} pN, {self.gmm.means[1]:.1f} pN\n"
            out += f"\tStd dev = {self.gmm.stds[0]:.2f} pN, {self.gmm.stds[1]:.2f} pN\n"
        else:
            out += f"GMM has not been fit\n"
        out += "\n"

        if self.psd:
            if self.psd_fit:
                out += f"PSD has been fit with:\n"
                out += f"\tAmplitude 1 = {self.psd_fit.A[0]:.2g} pN^2/Hz\n"
                out += f"\tCorner frequency 1 = {self.psd_fit.f_c[0]:.1f}  Hz\n"
                out += f"\tAmplitude 2 = {self.psd_fit.A[1]:.2g} pN^2/Hz\n"
                out += f"\tCorner frequency 2 = {self.psd_fit.f_c[1]:.0f} Hz\n"
            else:    
                out += "PSDs have been calculated but not fit\n"
        else:
            out += "PSDs have not been calculated\n"
        out += "\n"

        if self.wnhmm:
            out += "wnHMM has been fit with:\n"
            out += f"\tMeans: {self.wnhmm.fit_params.means[0]:.1f} pN, {self.wnhmm.fit_params.means[1]:.1f} pN\n"
            out += f"\tProbabilities: {self.wnhmm.fit_params.probabilities[0]:.1f} pN, {self.wnhmm.fit_params.probabilities[1]:.1f}"
            out += f"\tRates: {self.wnhmm.fit_params.rates[0]:.1f} Hz, {self.wnhmm.fit_params.rates[1]:.1f} Hz\n"
            out += f"\tStd devs: {np.sqrt(self.wnhmm.fit_params.variances[0]):.2f} pN, {np.sqrt(self.wnhmm.fit_params.variances[1]):.2f} pN\n"
            out += f"\tBIC: {(self.wnhmm.fit_quality.bic):.2g}\n"
            out += f"\tLog likelihood: {(self.wnhmm.fit_quality.loglikelihood):.2g}\n"
        else:
            out += "wnHMM has not been fit\n"
        out += "\n"
        
        if self.arhmm:
            out += "arHMM has been fit with:\n"
            out += f"\tMeans: {self.arhmm.fit_params.means[0]:.1f} pN, {self.arhmm.fit_params.means[1]:.1f} pN\n"
            out += f"\tProbabilities: {self.arhmm.fit_params.probabilities[0]:.1f} pN, {self.arhmm.fit_params.probabilities[1]:.1f}"
            out += f"\tRates: {self.arhmm.fit_params.rates[0]:.1f} Hz, {self.arhmm.fit_params.rates[1]:.1f} Hz\n"
            out += f"\tAlphas: {self.arhmm.fit_params.alphas[0]:.1f}, {self.arhmm.fit_params.alphas[1]:.1f}\n"
            out += f"\tStd devs: {np.sqrt(self.arhmm.fit_params.variances[0]):.2f} pN, {np.sqrt(self.arhmm.fit_params.variances[1]):.2f} pN\n"
            out += f"\tBIC: {(self.arhmm.fit_quality.bic):.2g}\n"
            out += f"\tLog likelihood: {(self.arhmm.fit_quality.loglikelihood):.2g}\n"
        else:
            out += "arHMM has not been fit\n"

        return out


    def plot_force_time(self, labels: np.ndarray|None=None, raw=False, ax: plt.Axes|None=None, **kwargs):
        """Make a force vs time plot.

        Args:
            labels (np.ndarray | None, optional): Labels to color data by. Defaults to None.
            raw (bool, optional): Whether to plot raw or downsampled data. Defaults to False.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            **kwargs: Additional arguments passed to plt.scatter.

        Returns:
            plt.Axes: Axes object containing plot.
        """
        ax = ax or plt.gca()
        
        if labels is not None:
            kwargs["c"] = ["tab:orange" if x else "tab:blue" for x in labels]

        if "s" not in kwargs:
            kwargs["s"] = 0.1

        if raw:
            force = self.raw_force[1:]
            time = self.raw_times[1:]
        else:
            force = self.force[1:]
            time = self.time[1:]
        
        ax.scatter(time, force, **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Force (pN)")

        return ax
    
    
    def plot_force_hist(self, binwidth=0.05, raw=False, ax: plt.Axes|None=None, **kwargs):
        """Make a force histogram.

        Args:
            binwidth (float, optional): Width of each force bin. Defaults to 0.05.
            raw (bool, optional): Whether to use raw or downsampled data. Defaults to False.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.

        Returns:
            plt.Axes: Axes object containing plot.
        """        
        ax = ax or plt.gca()

        if raw:
            force = self.raw_force
        else:
            force = self.force

        bins = np.arange(min(force), max(force) + binwidth, binwidth)
        ax.hist(force, bins=bins, density=True, **kwargs)
        ax.set_xlabel("Force (pN)")
        ax.set_ylabel("Probability density (pN$^{-1}$)")

        return ax
    

    def fit_gmm(self, probs_guess: np.ndarray, means_guess: np.ndarray, stds_guess: np.ndarray):        
        """Fit force histogram with 2 component GMM. By convention, the first state is lower force.

        Args:
            probs_guess (np.ndarray): Probabilities of occupying each state.
            means_guess (np.ndarray): Mean force of each state.
            stds_guess (np.ndarray): Standard deviation of force for each state.
        """        
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

        self.gmm = GMM_results(
            means=means,
            probabilities=probabilities,
            stds=np.sqrt(variances),
            variances=variances
        )


    def plot_gmm_pdf(self, annotate=True, ax: plt.Axes|None=None, line1_kwargs={}, line2_kwargs={}, line3_kwargs={}, **kwargs):
        """Plot the PDFs determined by fitting the GMM.

        Args:
            annotate (bool, optional): Whether to add values to plot. Defaults to True.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            line1_kwargs (dict, optional): Arguments passed to plt.plot for drawing state 1 pdf. Defaults to {}.
            line2_kwargs (dict, optional): Arguments passed to plt.plot for drawing state 2 pdf. Defaults to {}.
            line3_kwargs (dict, optional): Arguments passed to plt.plot for the line for the total pdf. Defaults to {}.

        Returns:
            plt.Axes: Axes object containing plot.
        """        
        ax = ax or plt.gca()

        gmm = self.gmm
        means = gmm.means
        stds = gmm.stds
        probs = gmm.probabilities

        forces = np.linspace(min(self.force), max(self.force), 100)
        forces = forces.reshape((-1, 1))

        pdfs = [probs[i]*norm(loc=means[i], scale=stds[i]).pdf(forces) for i in [0,1]]
        sum_pdf = pdfs[0]+pdfs[1]
        
        ax.plot(forces, pdfs[0], color="tab:blue", label="Peak 1", **line1_kwargs, **kwargs)
        ax.plot(forces, pdfs[1], color="tab:orange", label="Peak 2", **line2_kwargs, **kwargs)
        ax.plot(forces, sum_pdf, color="black", label="Total", **line3_kwargs, **kwargs)

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
    

    def calculate_psd(self, length=1.):
        """Calculate PSD of data using Welch's method.

        Args:
            length (float, optional): Length in seconds for each PSD. Defaults to 1..
        """
        f, psd = welch(self.force, self.f_s, nperseg=int(length*self.f_s))
        raw_f, raw_psd = welch(self.raw_force, self.raw_f_s, nperseg=int(length*self.raw_f_s))
        self.psd = PSD_results(f=f, value=psd)
        self.raw_psd = PSD_results(f=raw_f, value=raw_psd)

    
    def plot_psd(self, raw=False, log="loglog", ax: plt.Axes|None=None, **kwargs):
        """Plot PSD of the data.

        Args:
            raw (bool, optional): Whether to use raw or downsampled data. Defaults to False.
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.

        Returns:
            plt.Axes: Axes object containing plot.
        """
        ax = ax or plt.gca()

        if raw:
            data = self.raw_psd
        else:
            data = self.psd
        ax.plot(data.f, data.value, **kwargs)

        if log=="loglog":
            ax.loglog()
        elif log=="semilogx":
            ax.semilogx()
        elif log=="semilogy":
            ax.semilogy()
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (pN$^2$/Hz)")

        return ax
    

    def fit_psd(self, A_1_guess: float, f_c_1_guess: float, A_2_guess: float, f_c_2_guess: float, f_min=0., f_max=2e4, raw=False):
        """Fit PSD with an aliased double Lorentzian.

        Args:
            A_1_guess (float): Guess for amplitude of lower frequency peak in pN^2/Hz.
            f_c_1_guess (float): Guess for frequency of lower frequency peak in Hz.
            A_2_guess (float): Guess for amplitude of higher frequency peak in pN^2/Hz.
            f_c_2_guess (float): Guess for frequency of higher frequency peak in Hz.
            f_min (float, optional): Minimum frequency to fit. Defaults to 0.
            f_max (float, optional): Maximum frequency to fit. Defaults to 2e4.
            raw (bool, optional): Whether to fit raw data or downsampled data. Defaults to False.
        """
        if raw:
            psd = self.raw_psd
            f_s = self.raw_f_s
        else:
            psd = self.psd
            f_s = self.f_s
        f = psd.f
        value = psd.value
        mask = (f>f_min) & (f<f_max)
        f_mask = f[mask]
        value_mask = value[mask]

        guess = (A_1_guess, f_c_1_guess, A_2_guess, f_c_2_guess)

        fit_func = partial(aliased_double_lorentzian, f_s=f_s)

        fit, _ = curve_fit(fit_func, f_mask, value_mask, p0=guess, sigma=np.sqrt(value_mask))

        self.psd_fit = PSD_fit(A=np.array([fit[0], fit[2]]), f_c=np.abs([fit[1], fit[3]]))


    def plot_psd_fit(self, log="loglog", plot_all = True, raw = False, ax: plt.Axes|None=None, subplot1_kws: dict={}, subplot2_kws: dict={}, **kwargs):
        """Plots PSD fit.

        Args:
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            plot_all (bool, optional): If true, also plots individual Lorentzian fits. Defaults to True.
            raw (bool, optional): Whether to use raw data or downsampled data. Defaults to False.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            subplot1_kws (dict, optional): kwargs passed to plt.plot for the lower frequency Lorentzian. Defaults to {}.
            subplot2_kws (dict, optional): kwargs passed to plt.plot for the higher frequency Lorentzian. Defaults to {}.
            **kwargs: passed to plt.plot for the double Lorentzian.
        Returns:
            plt.Axes: Axes object containing plot.
        """             
        ax = ax or plt.gca()

        if raw:
            f = self.raw_psd["f"]
            f_s = self.raw_f_s
        else:
            f = self.psd["f"]
            f_s = self.f_s
        if plot_all:
            ax.plot(f, aliased_lorentzian(f, self.psd_fit.A[0], self.psd_fit.f_c[0], f_s), **subplot1_kws)
            ax.plot(f, aliased_lorentzian(f, self.psd_fit.A[1], self.psd_fit.f_c[1], f_s), **subplot2_kws)
        ax.plot(f, aliased_double_lorentzian(f, self.psd_fit.A[0], self.psd_fit.f_c[0], self.psd_fit.A[1], self.psd_fit.f_c[1], f_s), **kwargs)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (pN$^2$/Hz)")

        if log in ("loglog", "semilogx"):
            ax.set_xscale("log")

        if log in ("loglog", "semilogy"):
            ax.set_yscale("log")

        return ax
    

    @staticmethod
    def _calculate_lifetimes(labels, f_s):
        """
        transitions: last index before a state transition
        states: label of state before transition
        lengths: how long does state last until it transitions
        Folding is 0->1
        Unfolding is 1->0
        """

        delta = np.diff(labels)
        transitions = np.nonzero(delta)[0]

        states = labels[transitions]

        lengths = np.diff(np.insert(transitions, 0, -1))
        lifetimes = lengths/f_s

        unfolded_lts = lifetimes[states==0]
        folded_lts = lifetimes[states==1]

        folding_rate = 1/np.mean(unfolded_lts)
        unfolding_rate = 1/np.mean(folded_lts)
        rates = np.array([folding_rate, unfolding_rate])


        return LT_results(
            transitions=transitions,
            states=states,
            lengths=lengths,
            unfolded_lts=unfolded_lts,
            folded_lts=folded_lts,
            rates=rates
        )


    @staticmethod
    def _wnhmm_params_arr_to_obj(params: np.ndarray, f_s: float):
        # Unpack params array
        p_11, p_21 = params[0:2]
        means = params[2:4]
        variances = params[4:6]

        # Conversions
        k_1 = (1-p_11)*f_s
        k_2 = p_21*f_s
        rates = np.array([k_1, k_2])
        omega_c = np.array([k_1+k_2])
        probabilities = np.flip(rates)/omega_c
        f_c = omega_c/2/np.pi
        
        return HMM_params(
            means=means,
            variances=variances,
            probabilities=probabilities,
            f_cs=f_c,
            omega_cs=omega_c,
            rates=rates,
            alphas=np.array([]),
            transition_matrix=np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            params=params
        )
    

    def _wnhmm_get_default_params(self):
        f_s = self.f_s
        f_c = np.array(self.psd_fit.f_c[0]) 
        means = self.gmm.means
        probabilities = self.gmm.probabilities
        omega_c = 2*np.pi*f_c
        k_2, k_1 = probabilities*omega_c
        rates = np.array([k_1, k_2])
        p_11 = 1-k_1/f_s
        p_21 = k_2/f_s

        variances = self.gmm.variances
        params = np.array([p_11, p_21, *means, *variances])

        return HMM_params(
            means=means,
            variances=variances,
            probabilities=probabilities,
            f_cs=f_c,
            omega_cs=omega_c,
            rates=rates,
            alphas=np.array([]),
            transition_matrix=np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            params=params
        )


    def fit_wnhmm(self, params: np.ndarray|None=None):
        """Fit wnHMM.

        Args:
            params (np.ndarray | None, optional): Array containing [r_11, r_21, mean1, mean2, var1, var2].
                If None, computes from PSD and GMM fits. Defaults to None.
        """
        force = self.force

        if params is None:
            init_params = self._wnhmm_get_default_params()
        else:
            init_params = self._wnhmm_params_arr_to_obj(params, self.f_s)

        hmm = MarkovRegression(force, 2, switching_variance=True)
        hmm_fit = hmm.fit(init_params["params"])

        out_params = self._wnhmm_params_arr_to_obj(hmm_fit.params, self.f_s)

        labels = np.argmax(hmm_fit.smoothed_marginal_probabilities, axis=1)[1:]

        fit_quality = Fit_quality(
            standard_error=hmm_fit.bse,
            loglikelihood=hmm_fit.llf,
            bic=hmm_fit.bic,
            aic=hmm_fit.aic,
            hqic=hmm_fit.hqic
        )

        lifetimes = self._calculate_lifetimes(labels, self.f_s)

        self.wnhmm = HMM_results(
            initial_params=init_params,
            fit_params=out_params,
            fit_quality=fit_quality,
            labels=labels,
            smoothed_probabilities=hmm_fit.smoothed_marginal_probabilities,
            lifetimes=lifetimes
        )


    @staticmethod
    def _arhmm_params_arr_to_obj(params, f_s: float):
        p_11, p_21 = params[0:2]
        means = params[2:4]
        variances = params[4:6]
        alphas = params[6:8]

        k_1 = (1-p_11)*f_s
        k_2 = p_21*f_s

        rates = np.array([k_1, k_2])
        omega_c_1 = k_1+k_2
        probabilities = np.flip(rates)/omega_c_1
        omega_c_2 = -f_s*np.mean(np.log(alphas))

        omega_cs = np.array([omega_c_1, omega_c_2])
        f_cs = omega_cs/2/np.pi

        return HMM_params(
            means=means,
            variances=variances,
            probabilities=probabilities,
            f_cs=f_cs,
            omega_cs=omega_cs,
            rates=rates,
            alphas=alphas,
            transition_matrix=np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            params=params
        )
    

    def _arhmm_get_default_params(self):
        f_s = self.f_s
        f_cs = self.psd_fit.f_c
        omega_cs = f_cs*2*np.pi
        omega_c_1, omega_c_2 = omega_cs

        means = self.gmm.means
        probabilities = self.gmm.probabilities

        k_2, k_1 = probabilities*omega_c_1
        rates = np.array([k_1, k_2])
        p_11 = 1-k_1/f_s
        p_21 = k_2/f_s

        alpha = np.exp(-omega_c_2/f_s)
        alphas = np.array([alpha, alpha])

        variances = self.gmm["variances"]*(1-alphas**2)

        params = np.array([p_11, p_21, *means, *variances, *alphas])

        return HMM_params(
            means=means,
            variances=variances,
            probabilities=probabilities,
            f_cs=f_cs,
            omega_cs=omega_cs,
            rates=rates,
            alphas=alphas,
            transition_matrix=np.array([[p_11, p_21], [1-p_11, 1-p_21]]),
            params=params
        )


    def fit_arhmm(self, params: np.ndarray|None=None):
        """Fit arHMM.

        Args:
            params (np.ndarray | None, optional): Array containing [r_11, r_21, mean1, mean2, var1, var2, alpha1, alpha2].
                If None, computes from PSD and GMM fits. Defaults to None.
        """ 
        force = self.force

        if params is None:
            init_params = self._arhmm_get_default_params()
        else:
            init_params = self._arhmm_params_arr_to_obj(params, self.f_s)

        arhmm = MarkovAutoregression(force, 2, 1, switching_variance=True)
        arhmm_fit = arhmm.fit(init_params["params"])

        out_params = self._arhmm_params_arr_to_obj(arhmm_fit.params, self.f_s)

        labels = np.argmax(arhmm_fit.smoothed_marginal_probabilities, axis=1)

        fit_quality = Fit_quality(
            standard_error=arhmm_fit.bse,
            loglikelihood=arhmm_fit.llf,
            bic=arhmm_fit.bic,
            aic=arhmm_fit.aic,
            hqic=arhmm_fit.hqic
        )

        lifetimes = self._calculate_lifetimes(labels, self.f_s)

        self.arhmm = HMM_results(
            initial_params=init_params,
            fit_params=out_params,
            fit_quality=fit_quality,
            labels=labels,
            smoothed_probabilities=arhmm_fit.smoothed_marginal_probabilities,
            lifetimes=lifetimes
        )

    
    @staticmethod
    def _plot_lts(lifetimes: np.ndarray, ax: plt.Axes, fill: bool, log: str, **kwargs):
        ax = ax or plt.gca()

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
    

    def plot_folding_surv(self, ax: plt.Axes|None=None, fill=False, log: str|None=None, wnhmm=False, **kwargs):
        """Plot folding survival plot

        Args:
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            fill (bool, optional): Whether to fill the area under the curve. Defaults to False.
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            wnhmm (bool, optional): Whether to use lifetimes from wnHMM or arHMM. Defaults to False.

        Returns:
            plt.Axes: Axes object containing plot.
        """        
        ax = ax or plt.gca()

        if wnhmm:
            lifetimes = self.wnhmm.lifetimes.unfolded_lts
        else:
            lifetimes = self.arhmm.lifetimes.unfolded_lts

        ax = self._plot_lts(lifetimes, ax, fill, log, **kwargs)

        ax.set_ylabel("Fraction unfolded")

        return ax
    

    def plot_unfolding_surv(self, ax: plt.Axes|None=None, fill=False, log=None, hmm=False, **kwargs):
        """Plot unfolding survival plot

        Args:
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            fill (bool, optional): Whether to fill the area under the curve. Defaults to False.
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            wnhmm (bool, optional): Whether to use lifetimes from wnHMM or arHMM. Defaults to False.

        Returns:
            plt.Axes: Axes object containing plot.
        """
        ax = ax or plt.gca()

        if hmm:
            lifetimes = self.wnhmm.lifetimes.folded_lts
        else:
            lifetimes = self.arhmm.lifetimes.folded_lts

        ax = self._plot_lts(lifetimes, ax, fill, log, **kwargs)

        ax.set_ylabel("Fraction folded")

        return ax
    

    @staticmethod
    def _plot_rate(rate, time_lims: tuple, ax: plt.Axes, log: str, **kwargs):
        ax = ax or plt.gca()

        times = np.linspace(time_lims[0], time_lims[1], 200)

        ax.plot(times, np.exp(-rate*times), **kwargs)

        if log in ["loglog", "semilogx"]:
            ax.set_xscale("log")
        if log in ["loglog", "semilogy"]:
            ax.set_yscale("log")

        ax.set_xlabel("Time (s)")

        return ax


    def plot_folding_rate(self, from_lts=True, label_rate=True, min_time=0., max_time: float|None=None, ax: plt.Axes|None=None, log: str|None=None, wnhmm=False, **kwargs):
        """Plot folding survival from folding rate.

        Args:
            from_lts (bool, optional): Whether to use rate fit from lifetimes or from HMM. Defaults to True.
            label_rate (bool, optional): Whether to put rate into label. Defaults to True.
            min_time (float, optional): Minimum time to plot in s. Defaults to 0.
            max_time (float | None, optional): Maximum time to plot in s. If None, uses the maximum lifetime. Defaults to None.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            wnhmm (bool, optional): Whether to use lifetimes from wnHMM or arHMM. Defaults to False.

        Returns:
            _type_: _description_
        """        
        ax = ax or plt.gca()

        if wnhmm:
            model = self.wnhmm
        else:
            model = self.arhmm

        lts = model.lifetimes

        if from_lts:
            rate = lts.rates[0]
        else:
            rate = model.fit_params.rates[0]

        if max_time is None:
            max_time = np.max(lts.unfolded_lts)

        if label_rate:
            if from_lts:
                type_str = "lt"
            else:
                type_str = "mod"

            if wnhmm:
                model_str = "wnHMM"
            else:
                model_str = "arHMM"

            label = f"$k_{{\\mathrm{{{type_str},{model_str}}}}} = {rate:.1f}$ Hz"
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, label=label, **kwargs)
        else:
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, **kwargs)

        ax.set_ylabel("Fraction unfolded")
        
        return ax


    def plot_unfolding_rate(self, from_lts=True, label_rate=True, min_time=0., max_time: float|None=None, ax: plt.Axes|None=None, log: str|None=None, hmm=False, **kwargs):
        """Plot unfolding survival from folding rate.

        Args:
            from_lts (bool, optional): Whether to use rate fit from lifetimes or from HMM. Defaults to True.
            label_rate (bool, optional): Whether to put rate into label. Defaults to True.
            min_time (float, optional): Minimum time to plot in s. Defaults to 0.
            max_time (float | None, optional): Maximum time to plot in s. If None, uses the maximum lifetime. Defaults to None.
            ax (plt.Axes | None, optional): Axis to plot data on. If none, uses the current axis. Defaults to None.
            log (str, optional): Log scaling of the plot. Defaults to "loglog".
            wnhmm (bool, optional): Whether to use lifetimes from wnHMM or arHMM. Defaults to False.

        Returns:
            _type_: _description_
        """        
        ax = ax or plt.gca()

        if hmm:
            model = self.wnhmm
        else:
            model = self.arhmm

        lts = model.lifetimes
        if from_lts:
            rate = lts.rates[1]
        else:
            rate = model.fit_params.rates[1]

        if max_time is None:
            max_time = np.max(lts.folded_lts)

        if label_rate:
            if from_lts:
                type_str = "lt"
            else:
                type_str = "mod"

            if hmm:
                model_str = "wnHMM"
            else:
                model_str = "arHMM"

            label = f"$k_{{\\mathrm{{{type_str},{model_str}}}}} = {rate:.1f}$ Hz"
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, label=label, **kwargs)
        else:
            ax = self._plot_rate(rate, (min_time, max_time), ax, log, **kwargs)

        ax.set_ylabel("Fraction folded")

        return ax


def from_lk(filename: Path | str, downsample: int=2):
    """Load and downsample data from a Lumicks h5 file.

    Args:
        filename (Path | str): Lumicks h5 file location.
        downsample (int, optional): Downsampling factor. Defaults to 2.

    Returns:
        Switching: Object of class Switching.
    """
    filename = str(filename)
    lk_file = lk.File(filename)

    raw_time = lk_file.force1x.seconds
    raw_force = (lk_file.force2x.data-lk_file.force1x.data)/2
    raw_f_s = lk_file.force1x.sample_rate

    return Switching(
        file = filename,
        raw_time = raw_time,
        raw_force = raw_force,
        raw_f_s = raw_f_s,
        raw_N = len(raw_time),
        downsample = downsample,
        time = raw_time[::downsample],
        force = raw_force[::downsample],
        f_s = raw_f_s/downsample,
        N = len(raw_time[::downsample])
    )


def from_data(force: np.ndarray, time: np.ndarray, f_s: float, downsample: int=1):
    """Load and downsample data from arrays.

    Args:
        force (np.ndarray): Force array in pN.
        time (np.ndarray): Time array in s.
        f_s (float): Sample rate in Hz.
        downsample (int, optional): Downsampling factor. Defaults to 1.

    Returns:
        Switching: Object of class Switching.
    """ 
    return Switching(
        file = "",
        raw_time = time,
        raw_force = force,
        raw_f_s = f_s,
        raw_N = len(time),
        downsample = downsample,
        time = time[::downsample],
        force = force[::downsample],
        f_s = f_s/downsample,
        N = len(time[::downsample])
    )

