import numpy as np
import matplotlib.pyplot as plt


def generate_ar1_data(N: int, mean: float, sigma: float, alpha: float, seed: int) -> np.ndarray:
    """Generate a sequence using the AR(1) model.

    Args:
        N (int): Number of points to generate.
        mean (float): The mean value of the distribution.
        sigma (float): The standard deviation of the distribution. Note the standard deviation of the noise term is sigma*sqrt(1-alpha^2).
        alpha (float): The autocorrelation of the distribution.

    Returns:
        ndarray: Sequence of N points with mean=mean, standard deviation=sigma, autocorrelation=alpha
    """    
    x = np.zeros(int(N))
    
    sigma_eta = sigma*np.sqrt(1-alpha**2)

    rng = np.random.default_rng(seed)
    r = rng.normal(0, sigma_eta, N-1)

    x[0] = mean+rng.normal(0, sigma)
    for i in range(1, N):
        x[i] = (1-alpha)*mean + alpha*x[i-1] + r[i-1]

    return x


def generate_independent_data(N: int, mean: float, sigma: float, seed: int) -> np.ndarray:
    """Generate a sequence of uncorrelated Gaussian data.

    Args:
        N (int): Number of points to generate.
        mean (float): The mean value of the distribution.
        sigma (float): The standard deviation of the distribution.

    Returns:
        np.ndarray: Sequence of N points with mean=mean, standard deviation=sigma, and no autocorrelation.
    """    
    rng = np.random.default_rng(seed)
    return rng.normal(mean, sigma, N)


def get_runs(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find runs of consecutive states.

    Args:
        states (np.ndarray): The states to search through.

    Returns:
        transitions (np.ndarray): The indices of the last element before transitioning to a new state, including the last state at the end of states.
        start_states (np.ndarray): The state of each run.
        lengths (np.ndarray): The length of each run.
    """    
    delta = np.diff(states)
    transitions = np.nonzero(delta)[0]
    transitions = np.append(transitions, len(states)-1)

    start_states = states[transitions]

    lengths = np.diff(transitions)
    lengths = np.insert(lengths, 0, transitions[0]+1)
    return transitions, start_states, lengths


def generate_hmm_states(N: int, rates: list, seed: int|None =None) -> np.ndarray:
    """Generate sequence from a 2 state Markov chain.

    Args:
        N (int): Length of sequence to generate
        rates (list): Transition rates from state 0->1 and 1->0.

    Returns:
        np.ndarray: States of N points. States are 0 or 1. Transition rates are determined by rates. 
    """    
    states = np.zeros(N, int)

    # Probability of starting in state 0
    p0 = rates[1]/(rates[0]+rates[1])

    rng = np.random.default_rng(seed)
    probs = rng.uniform(size=N)

    states[0] = int(probs[0]>p0)
    for i in range(1, N):
        if probs[i] > rates[states[i-1]]:
            states[i] = states[i-1]
        else:
            states[i] = 1 -  states[i-1]
    
    return states


def generate_arhmm_data(states: np.ndarray, means: list, sigmas: list, alphas: list, seed: int|None = None) -> np.ndarray:
    """Generate autoregressive data for input state sequence.

    Args:
        states (np.ndarray): States of Markov chain.
        means (list): Contains means of different states.
        sigmas (list): Contains standard deviations of different states.
        alphas (list): Contains autocorrelation of different states.

    Returns:
        np.ndarray: Data generated with autoregressive parameters for given input states.
    """    
    transitions, start_states, lengths = get_runs(states)
    n_points = len(states)

    n_segments = len(transitions)

    out = np.zeros(n_points)
    for i in range(n_segments):
        state = int(start_states[i])
        length = lengths[i]

        if i==0:
            start = 0
        else:
            start = transitions[i-1]+1
        end = start+length

        out[start:end] = generate_ar1_data(length, means[state], sigmas[state], alphas[state], seed)
        if seed is not None:
            seed += 1

    return out


def generate_independent_hmm_data(states: np.ndarray, means: list, sigmas: list, seed: int) -> np.ndarray:
    """Generate independent Gaussian data for input state sequence.

    Args:
        states (np.ndarray): States of Markov chain.
        means (list): Contains means of different states.
        sigmas (list): Contains standard deviations of different states.

    Returns:
        np.ndarray: Data generated with Gaussian parameters for given input states.
    """    
    transitions, start_states, lengths = get_runs(states)
    n_points = len(states)

    n_segments = len(transitions)

    out = np.zeros(n_points)
    for i in range(n_segments):
        state = int(start_states[i])
        length = lengths[i]

        if i==0:
            start = 0
        else:
            start = transitions[i-1]+1
        end = start+length
        out[start:end] = generate_independent_data(length, means[state], sigmas[state], seed+i)

    return out

def E_step(states, observations, rates, means, sigmas):
    k01 = rates[0]
    k10 = rates[1]
    k00 = 1 - k01
    k11 = 1 - k10
    
    p0 = k10/(k10+k01)
    p1 = 1 - p0

    N = len(states)

    alpha0 = np.zeros(N)
    alpha1 = np.zeros(N)
    beta0 = np.zeros(N)
    beta1 = np.zeros(N)
    ll0 = np.zeros(N)
    ll1 = np.zeros(N)

    c = np.zeros(N)

    alpha0[0] = p0*ll0[0]
    alpha1[0] = p1*ll1[0]
    c[0] = alpha0[0]+alpha1[0]

    alpha0[0] /= c[0]
    alpha1[0] /= c[0]


    for i in range(1, N):
        alpha0[i] = ll0[i]*(alpha0[i-1]*k00 + alpha1[i-1]*k10)
        alpha1[i] = ll1[i]*(alpha0[i-1]*k01 + alpha1[i-1]*k11)

        c[i] = alpha0[i] + alpha1[i]
        
        alpha0[i] /= c[i]
        alpha1[i] /= c[i]
    
    beta0[-1] = 1
    beta1[-1] = 1

    for i in range(N-2, -1, -1):
        b0 = beta0[i+1]*ll0[i+1]/c[i+1]
        b1 = beta1[i+1]*ll1[i+1]/c[i+1]
        beta0[i] = b0*k00 + b1*k01
        beta1[i] = b0*k10 + b1*k11

    gamma0 = alpha0*beta0
    gamma1 = alpha1*beta1

    eta0 = c[1:]*beta0[1:]
    eta1 = c[1:]*beta1[1:]

    eta00 = eta0*alpha0[:-1]*k00
    eta01 = eta1*alpha0[:-1]*k01
    eta10 = eta0*alpha1[:-1]*k10
    eta11 = eta1*alpha1[:-1]*k11

    #A00 = np.sum(eta)

def fit_ar1(data):
    N = len(data)

    # Calculate sums from 0 to N-2 and 1 to N-1
    sum_inner = np.sum(data[1:-1])
    sum_start = sum_inner+data[0]
    sum_end = sum_inner+data[-1]

    # Calculate the sum of data^2 from 1 to N-1
    sum_squared = np.dot(data[1:], data[1:])

    # Calculate the 1-lag autocorrelation, the sum of data_i-1 * data_i from 1 to N-1 
    sum_prod = np.dot(data[:-1], data[1:])

    # Normalization constant from determinant in matrix inverse
    C = (N-1)*sum_squared - sum_start**2

    # Calculate MLE for AR1 parameters
    alpha = ((N-1)*sum_prod - sum_start*sum_end) / C
    mean = (sum_squared*sum_end - sum_start*sum_prod)/(1-alpha)/C
    sigma = np.linalg.norm(data[1:]-mean*(1-alpha)-alpha*data[:-1])/np.sqrt((N-1)*(1-alpha**2))

    return mean, sigma, alpha

def fit_gauss(data):
    return np.mean(data), np.std(data)


def gauss_ll(data, mean, sigma):
    return -np.sum((data-mean)**2)/(2*sigma**2) - len(data)*np.log(sigma*np.sqrt(2*np.pi))

def ar_ll(data, mean, sigma, alpha):
    sigma_eta = sigma*np.sqrt(1-alpha**2)
    return -np.sum((data[1:]-alpha*data[:-1]-mean*(1-alpha))**2)/(2*sigma_eta**2) - (len(data)-1)*np.log(sigma_eta*np.sqrt(2*np.pi)) - (data[0]-mean)**2/(2*sigma**2) - np.log(sigma*np.sqrt(2*np.pi))

def exp_ll(N, sigma, alpha):
    sigma_eta = sigma*np.sqrt(1-alpha**2)
    gauss =  -N/2*(1 + np.log(2*np.pi) + np.log(sigma**2))
    ar = -N/2*(1 + np.log(2*np.pi) + np.log(sigma_eta**2))
    ar_on_gauss = -N/2*((1+alpha**2)/(1-alpha**2) + np.log(2*np.pi) + np.log(sigma_eta**2))
    return gauss, ar, ar_on_gauss


def plot_hist(x, bins=100, label=None, ax=None, orientation="vertical", histtype="step", color=None):
    if ax is None:
        ax = plt.gca()
    
    ax.hist(x, bins=bins, density=True, histtype=histtype, label=label, orientation=orientation, color=color)


def calculate_lifetimes(labels, sample_rate):
    # 1 == folded, 0 == unfolded
    last = labels[0]
    last_pos = 0
    zeros = []
    ones = []
    for i, x in enumerate(labels):
        if x != last:
            length = i - last_pos
            if last == 1:
                ones.append(length)
            else:
                zeros.append(length)
            last = x
            last_pos = i
    ones = np.array(sorted(ones))/sample_rate
    zeros = np.array(sorted(zeros))/sample_rate

    return ones, zeros

def calculate_rates(folded_lt, unfolded_lt):
    out = {}
    out["Unfolding rate"] = 1/folded_lt.mean()
    out["Unfolding rate sd"] = out["Unfolding rate"]/np.sqrt(len(folded_lt))

    out["Folding rate"] = 1/unfolded_lt.mean()
    out["Folding rate sd"] = out["Folding rate"]/np.sqrt(len(unfolded_lt))

    return out

def plot_survival(lifetime, params, folded=True, ax=None):
    if ax is None:
        ax = plt.gca()

    if folded:
        state = "folded"
        transition = "Unfolding"
    else:
        state = "unfolded"
        transition = "Folding"
    
    rate = params[f"{transition} rate"]
    sd = params[f"{transition} rate sd"]

    survival = np.flip(np.linspace(0, 1, num=len(lifetime)))
    ts = np.linspace(lifetime[0], lifetime[-1])
    
    ax.plot(lifetime, survival, drawstyle='steps-post')
    ax.plot(ts, np.exp(-rate*ts), 'k')
    ax.fill_between(ts, np.exp(-(rate+sd)*ts), np.exp(-(rate-sd)*ts), facecolor="tab:orange", alpha=0.25)
    
    ax.semilogy()
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"Fraction {state}")
    ax.set_title(f"{transition} rate {rate:.2f} $\\pm$ {sd:.2f} Hz")

    return ax


def simulatate_ARHMM(n, params, seed=None):
    p_11, p_21, mu_1, mu_2, sigma_2, alpha_1, alpha_2 = params
    p_switch = np.array([1-p_11, p_21])
    p_1 = p_21/np.sum(p_switch)

    mus = np.array([mu_1, mu_2])

    rng = np.random.default_rng(seed=seed)
    r_init = rng.random(1)
    r_switch = rng.random(n-1)
    r_value = rng.normal(0, np.sqrt(sigma_2), n)

    states = np.zeros(n, dtype=int)
    values = np.zeros(n)

    if r_init > p_1:
        states[0] = 1
    
    for i, r in enumerate(r_switch):
        if r < p_switch[states[i]]:
            states[i+1] = 1-states[i]
        else:
            states[i+1] = states[i]
    
    values = r_value +mus[states]

    return states, values