"""Harmonic balance linearisation

Rick Lupton
27 March 2014
"""

from __future__ import division
import numpy as np
from numpy import pi, cos, dot, array, fft
from scipy.integrate import complex_ode
from scipy.misc import derivative
import matplotlib.pyplot as plt


def get_harmonics(y):
    """Get first two Fourier harmonics from y
    (mean and first harmonic)
    """
    y = np.atleast_2d(y)
    assert y.shape[1] % 2 == 0, "should be even length"
    Y = fft.rfft(y, axis=1) / y.shape[1]

    # Check for aliasing
    #scale = abs(Y[:, 1:4]).max()
    np.testing.assert_allclose(Y[:, -1], 0, atol=abs(Y[:, :4]).max() * 1e-2)
    f0 = np.real(Y)[:, 0]
    fb = Y[:, 1] * 2
    return f0, fb


def from_harmonics(harmonics, n):
    """Return timeseries of function and its derivative"""
    assert n % 2 == 0, "should be even length"
    x0, xb = harmonics
    Y = np.c_[x0, xb / 2] * n
    return fft.irfft(Y, n, axis=1)


def calc_first_harmonic(func, x0, xb, w=None, n=512):
    """Calculate the first Fourier harmonic of func, given harmonic input.

    Parameters
    ----------
    func: callable
        The function to be linearised: func(x [, xdot]).
        Parameters `x` and `xdot` are arrays and func should return an array
        of the same size. `xdot' is only passed if `w' is given.
    x0: array
        Mean value of harmonic input
    xb: array
        Amplitude of harmonic input
    w: float [optional]
        Angular frequency (used for calculating xdot)
    """
    x0, xb = np.asarray(x0), np.asarray(xb)
    x = from_harmonics((x0, xb), n)
    if w is None:
        f = func(x)
    else:
        xdot = from_harmonics((0*x0, 1j*w*xb), n)
        f = func(x, xdot)
    f0, fb = get_harmonics(f)
    return f0, fb


def _freq_bins(freqs, n_periods):
    dw = np.min(freqs) / n_periods
    i1 = int(freqs[0] / dw)
    i2 = int(freqs[1] / dw)
    return i1, i2


def get_harmonics3(y, freqs, n_periods):
    """Get first two Fourier harmonics from y
    (mean and first harmonic)
    """
    w1, w2 = freqs
    y = np.atleast_2d(y)
    assert y.shape[1] % 2 == 0, "should be even length"
    window = np.ones(y.shape[1])
    Y = fft.rfft(window[np.newaxis, :] * y, axis=1) / np.sum(window)

    i1, i2 = _freq_bins(freqs, n_periods)

    # Check for aliasing
    scale = max(1e-10, abs(Y[:, :4]).max())
    np.testing.assert_allclose(Y[:, -1], 0, atol=scale * 1e-2)
    f0 = np.real(Y)[:, 0]
    if i1 == i2:
        f1 = Y[:, i1] * 2   # arbitarily choose w1
        f2 = 0
    else:
        f1 = Y[:, i1] * 2
        f2 = Y[:, i2] * 2
    return np.c_[f0, f1, f2]


def from_harmonics3(harmonics, freqs, n_per_period, n_periods):
    """Return timeseries of function and its derivative"""
    assert n_per_period % 2 == 0, "should be even length"
    harmonics = np.atleast_2d(harmonics)

    n = n_per_period * n_periods
    assert n % 2 == 0, "should be even length"
    i1, i2 = _freq_bins(freqs, n_periods)

    # Make Fourier coefficient matrix
    Y = np.zeros((harmonics.shape[0], n//2 + 1), dtype=np.complex)
    Y[:, 0] = harmonics[:, 0]
    Y[:, i1] += harmonics[:, 1] / 2
    Y[:, i2] += harmonics[:, 2] / 2

    return fft.irfft(Y * n, n, axis=1)


def _calc_mixed_harmonics(func, harmonics, freqs,
                          n_per_period, n_lf_periods):
    x = from_harmonics3(harmonics, freqs, n_per_period, n_lf_periods)
    if freqs is None:
        f = func(x)
    else:
        dot_harmonics = harmonics.copy()
        for i, w in enumerate([0, freqs[0], freqs[1]]):
            dot_harmonics[:, i] *= 1j * w
        xdot = from_harmonics3(dot_harmonics, freqs, n_per_period,
                               n_lf_periods)
        f = func(x, xdot)
    hf = get_harmonics3(f, freqs, n_lf_periods)
    return hf


def calc_mixed_harmonics(func, xs, ws=None, n_lf_periods=1,
                         n_per_hf_period=256, check_convergence=True):
    """Calculate the first Fourier harmonic of func, given harmonic input.

    Parameters
    ----------
    func: callable
        The function to be linearised: func(x [, xdot]).
        Parameters `x` and `xdot` are arrays and func should return an array
        of the same size. `xdot' is only passed if `w' is given.
    xs: array, shape (nvars, 3) or (3,)
        Harmonic input: mean value, amplitude at w1, amplitude at w2
    ws: array, shape (2,) [optional]
        Angular frequencies [w1, w2] (used for calculating xdot)
    """
    xs = np.atleast_2d(xs).astype(np.complex)

    # Calculate minimum number of points per period
    n_per_lf_period = int(n_per_hf_period * np.max(ws) / np.min(ws) / 2) * 2

    # starting number of periods
    if ws is not None and (abs(ws[0] - ws[1]) != 0 and
                           abs(2*ws[0] - ws[1]) != 0 and
                           abs(ws[0] - 2*ws[1]) != 0):
        num_periods_required = [
            int(np.ceil(np.min(ws) / abs(ws[0] - ws[1]) / 2) * 2),
            int(np.ceil(np.min(ws) / abs(2*ws[0]-ws[1]) / 2) * 2),
            int(np.ceil(np.min(ws) / abs(ws[0]-2*ws[1]) / 2) * 2),
        ]
        n = np.max(num_periods_required)
        if n > n_lf_periods:
            print("Small frequency gap ({}), incr num LF periods {} -> {}"
                  .format(np.argmax(num_periods_required), n_lf_periods, n))
            if n > 1000:
                raise RuntimeError("Too many periods required")
            n_lf_periods = n

    hf = _calc_mixed_harmonics(func, xs, ws, n_per_lf_period, n_lf_periods)
    print("periods: {}  {: .2f}   {: .2f}   {: .2f}"
          .format(n_lf_periods, *hf[0]))

    if not check_convergence:
        return hf

    while n_lf_periods < 1000:
        # Check convergence
        n_lf_periods *= 2
        hf_test = _calc_mixed_harmonics(func, xs, ws, n_per_lf_period,
                                        n_lf_periods)
        print("periods: {}  {: .2f}   {: .2f}   {: .2f}"
              .format(n_lf_periods, *hf_test[0]))
        if all(np.allclose(a, b, rtol=1e-3, atol=1e-6)
               for a, b in zip(hf.T, hf_test.T)):
            return hf_test
        else:
            hf = hf_test

    raise RuntimeError("Not converged after {} periods"
                       .format(n_lf_periods))


# def calc_first_harmonic(func, x0, xb, w=1.0):
#     """Calculate the first Fourier harmonic of func, given harmonic input.

#     Parameters
#     ----------
#     func: callable
#         The function to be linearised: func(x, xdot).
#         Parameters `x` and `xdot` are arrays and func should return an array.
#     x0: array
#         Mean value of harmonic input
#     xb: array
#         Amplitude of harmonic input
#     w: float [default 1.0]
#         Angular frequency (used for calculating xdot)
#     """

#     # Harmonic input function x, as a function of theta in (0, 2*pi)
#     x0 = np.asarray(x0)
#     xb = np.asarray(xb)
#     x = lambda theta: x0 + np.real(xb * np.exp(1j * theta))
#     xdot = lambda theta: np.real(1j * w * xb * np.exp(1j * theta))

#     # Find out how many outputs it has
#     test = func(x0, 0*x0)
#     unwrap = False
#     if isinstance(test, np.ndarray) and test.ndim > 0:
#         N = len(test)
#         wrapped_func = func
#     elif isinstance(test, (tuple, list)):
#         N = len(test)
#         wrapped_func = lambda x, xdot: array(func(x, xdot))
#     else:
#         # Wrap output in array
#         wrapped_func = lambda x, xdot: array([func(x, xdot)])
#         N = 1
#         unwrap = True

#     # Define integrand
#     def f(theta, y):
#         fval = wrapped_func(x(theta), xdot(theta))
#         return np.concatenate([fval, fval * np.exp(-1j * theta)])

#     # Integrate!
#     integrator = complex_ode(f)                    \
#         .set_integrator('dopri5', nsteps=1000, atol=1e-5, rtol=1e-4)     \
#         .set_initial_value(np.zeros(2*N, complex))
#     integrator.integrate(2*pi)

#     # Extract results
#     f0 = np.real(integrator.y[:N]) / (2*pi)
#     fb = integrator.y[N:] / pi

#     if unwrap:
#         f0, fb = f0[0], fb[0]
#     return f0, fb


def calc_first_harmonic_from_data(y, w=1.0):
    """Calculate the first Fourier harmonic of the function f(t)

    Parameters
    ----------
    y: array
        Evaluations of f(t) at equally-spaced time points
    w: float [default 1.0]
        Angular frequency (used for calculating xdot)
    """

    N = len(y)
    theta = np.linspace(0, 2 * pi, N, endpoint=False)
    exp = np.exp(-1j * theta)
    while exp.ndim < y.ndim:
        exp = exp[:, newaxis]

    f0 = np.trapz(y, axis=0) / N
    fb = np.trapz(y * exp, axis=0) / N * 2

    return f0, fb


def one(n, i):
    y = np.zeros(n)
    y[i] = 1
    return y


def calc_tangent_stiffness(func, x0, dx=0.5):
    N = len(x0)
    def perturb1(x, i):
        return func(x0 + x*one(N, i), 0*x0)
    def perturb2(x, i):
        return func(x0, x*one(N, i))
    K = np.array([derivative(perturb1, 0, dx, args=(i,)) for i in range(N)]).T
    C = np.array([derivative(perturb2, 0, dx, args=(i,)) for i in range(N)]).T
    f0 = func(x0, 0*x0)
    return f0, K, C


def calc_first_harmonic_tangent(func, x0, xb, w=1.0, dx=0.5):
    """This is relatively meaningless - most of the time want
    `calc_tangent_stiffness' instead. This function matches
    `calc_first_harmonic' though so is sometimes useful."""
    f0, K, C = calc_tangent_stiffness(func, x0, dx)
    fb = dot(K, xb) + 1j*w*dot(C, xb)
    return f0, fb


def plot_linearisation_diagnostics(x0, xb, f0, fb, f0T, fbT, func, label=''):
    # Check linearisation really is optimum
    th = np.linspace(0, 2*pi, 50)
    x = lambda theta: x0 + abs(xb)*cos(theta + np.angle(xb))
    Fest = lambda f0, fb: f0 + abs(fb)*cos(th + np.angle(fb))
    Fact = np.vectorize(func)(x(th))
    Ferr = lambda f0, fb: np.sqrt(np.trapz((Fest(f0, fb) - Fact)**2, th) /
                                  np.trapz(Fact**2, th))

    if label:
        print("Error for {}:".format(label))
    print(" answer: {:.3%}" .format(Ferr(f0*1.00, fb*1.00)))
    print(" +1% f0: {:+.3%}".format(Ferr(f0*1.01, fb*1.00) - Ferr(f0, fb)))
    print(" -1% f0: {:+.3%}".format(Ferr(f0*0.99, fb*1.00) - Ferr(f0, fb)))
    print(" +1% K : {:+.3%}".format(Ferr(f0*1.00, fb*1.01) - Ferr(f0, fb)))
    print(" -1% K : {:+.3%}".format(Ferr(f0*1.00, fb*0.99) - Ferr(f0, fb)))

    xs = np.linspace(max(0, x0-abs(xb)), x0+abs(xb), 50)
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    ax[0].plot(x(th), Fact, 'g')
    ax[0].plot(x(th), Fest(f0T, fbT), 'y', lw=1)
    ax[0].plot(x(th), Fest(f0, fb), 'k', lw=2)
    ax[0].axvline(x0, c='k', ls=':')
    ax[0].axhline(f0, c='k', ls=':')
    ax[0].set_title(label)
    ax[1].plot(th, Fact, 'g')
    ax[1].plot(th, f0T + fbT.real*cos(th), 'y', lw=1)
    ax[1].plot(th, f0 + fb.real*cos(th), 'k', lw=2)
    ax[1].axhline(f0, c='k', ls=':')
    ax[1].set_title("Error: {:.1%}  ({:.1%} T)"
                    .format(Ferr(f0, fb), Ferr(f0T, fbT)))
    plt.show()
