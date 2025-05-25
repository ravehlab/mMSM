"""
--------------------------------------------------------------------------------------------
This file contains code extracted and modified from MSMTools to avoid creating an explicit
dependency and to allow specifying an initial guess for the eigenvector in
stationary_distribution_from_backward_iteration.

Original source: https://github.com/markovmodel/msmtools

Modifications:
    - Minor interface change: allow initial guess argument.
    - Suppressed certain warnings.
    - Removed support for reversible matrices in eigenvalues function.

Copyright (c) 2014-2015 Computational Molecular Biology Group,
Freie Universitaet Berlin (GER)

This file is part of MSMTools and is licensed under the GNU Lesser General Public License
v3.0 (LGPL-3.0) or (at your option) any later version.

You should have received a copy of the GNU Lesser General Public License along with this file.
If not, see <https://www.gnu.org/licenses/>.

See COPYING.LESSER in the source directory for the full license text.
--------------------------------------------------------------------------------------------
"""



r"""This module provides functions for the coputation of stationary
vectors of stochastic matrices

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from numpy.linalg import eig, eigvals


def backward_iteration(A, mu, x0, tol=1e-14, maxiter=100):
    r"""Find eigenvector to approximate eigenvalue via backward iteration.

    Parameters
    ----------
    A : (N, N) ndarray
        Matrix for which eigenvector is desired
    mu : float
        Approximate eigenvalue for desired eigenvector
    x0 : (N, ) ndarray
        Initial guess for eigenvector
    tol : float
        Tolerace parameter for termination of iteration

    Returns
    -------
    x : (N, ) ndarray
        Eigenvector to approximate eigenvalue mu

    """
    T = A - mu * np.eye(A.shape[0])
    """LU-factor of T"""
    lupiv = lu_factor(T)
    """Starting iterate with ||y_0||=1"""
    r0 = 1.0 / np.linalg.norm(x0)
    y0 = x0 * r0
    """Local variables for inverse iteration"""
    y = 1.0 * y0
    r = 1.0 * r0
    for i in range(maxiter):
        x = lu_solve(lupiv, y)
        r = 1.0 / np.linalg.norm(x)
        y = x * r
        if r <= tol:
            return y
    msg = "Failed to converge after %d iterations, residuum is %e" % (maxiter, r)
    raise RuntimeError(msg)


def stationary_distribution_from_backward_iteration(P, x0=None, eps=1e-15):
    r"""Fast computation of the stationary vector using backward
    iteration.

    Parameters
    ----------
    P : (M, M) ndarray
        Transition matrix
    eps : float (optional)
        Perturbation parameter for the true eigenvalue.

    Returns
    -------
    pi : (M,) ndarray
        Stationary vector

    """
    A = np.transpose(P)
    mu = 1.0 - eps
    if x0 is None:
        x0 = np.ones(P.shape[0])
    y = backward_iteration(A, mu, x0)
    pi = y / y.sum()
    return pi


def stationary_distribution_from_eigenvector(T):
    r"""Compute stationary distribution of stochastic matrix T.

    The stationary distribution is the left eigenvector corresponding to the
    non-degenerate eigenvalue :math: `\lambda=1`.

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    val, L = eig(T.T)

    """ Sorted eigenvalues and left and right eigenvectors. """
    perm = np.argsort(val)[::-1]

    val = val[perm]
    L = L[:, perm]
    """ Make sure that stationary distribution is non-negative and l1-normalized """
    nu = np.abs(L[:, 0])
    mu = nu / np.sum(nu)
    return mu


def stationary_distribution(T, x0=None):
    r"""Compute stationary distribution of stochastic matrix T.

    Chooses the fastest applicable algorithm automatically

    Input:
    ------
    T : numpy array, shape(d,d)
        Transition matrix (stochastic matrix).

    Returns
    -------
    mu : numpy array, shape(d,)
        Vector of stationary probabilities.

    """
    fallback = False
    try:
        mu = stationary_distribution_from_backward_iteration(T, x0)
        if np.any(mu < 0):  # numerical problem, fall back to more robust algorithm.
            fallback=True
    except (RuntimeError, ValueError) as e:
        fallback = True

    if fallback:
        mu = stationary_distribution_from_eigenvector(T)
        if np.any(mu < 0):  # still? Then set to 0 and renormalize
            mu = np.maximum(mu, 0.0)
            mu /= mu.sum()

    return mu

def timescales(T, tau=1, k=None, reversible=False, mu=None):
    r"""Compute implied time scales of given transition matrix

    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    tau : int, optional
        lag time
    k : int, optional
        Number of time scales
    reversible : bool, optional
        Indicate that transition matirx is reversible
    mu : (M,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    ts : (N,) ndarray
        Implied time scales of the transition matrix.
        If k=None then N=M else N=k

    Notes
    -----
    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    values = eigenvalues(T, reversible=reversible, mu=mu)

    """Sort by absolute value"""
    ind = np.argsort(np.abs(values))[::-1]
    values = values[ind]

    if k is None:
        values = values
    else:
        values = values[0:k]

    """Compute implied time scales"""
    return timescales_from_eigenvalues(values, tau)


def timescales_from_eigenvalues(evals, tau=1):
    r"""Compute implied time scales from given eigenvalues

    Parameters
    ----------
    evals : eigenvalues
    tau : lag time

    Returns
    -------
    ts : ndarray
        The implied time scales to the given eigenvalues, in the same order.

    """

    """Check for dominant eigenvalues with large imaginary part"""

    # if not np.allclose(evals.imag, 0.0):
    #     warnings.warn('Using eigenvalues with non-zero imaginary part', ImaginaryEigenValueWarning)

    """Check for multiple eigenvalues of magnitude one"""
    ind_abs_one = np.isclose(np.abs(evals), 1.0, rtol=0.0, atol=1e-14)
    # if sum(ind_abs_one) > 1:
    #     warnings.warn('Multiple eigenvalues with magnitude one.', SpectralWarning)

    """Compute implied time scales"""
    ts = np.zeros(len(evals))

    """Eigenvalues of magnitude one imply infinite timescale"""
    ts[ind_abs_one] = np.inf

    """All other eigenvalues give rise to finite timescales"""
    ts[np.logical_not(ind_abs_one)] = \
        -1.0 * tau / np.log(np.abs(evals[np.logical_not(ind_abs_one)]))
    return ts

def eigenvalues(T, k=None, reversible=False, mu=None):
    r"""Compute eigenvalues of given transition matrix.

    Parameters
    ----------
    T : (d, d) ndarray
        Transition matrix (stochastic matrix)
    k : int or tuple of ints, optional
        Compute the first k eigenvalues of T
    reversible : bool, optional
        Indicate that transition matrix is reversible
    mu : (d,) ndarray, optional
        Stationary distribution of T

    Returns
    -------
    eig : (n,) ndarray,
        The eigenvalues of T ordered with decreasing absolute value.
        If k is None then n=d, if k is int then n=k otherwise
        n is the length of the given tuple of eigenvalue indices.

    Notes
    -----
    Eigenvalues are computed using the numpy.linalg interface
    for the corresponding LAPACK routines.

    If reversible=True the the eigenvalues of the similar symmetric
    matrix `\sqrt(\mu_i / \mu_j) p_{ij}` will be computed.

    The precomputed stationary distribution will only be used if
    reversible=True.

    """
    # if reversible:
    #     try:
    #         evals = eigenvalues_rev(T, k=k, mu=mu)
    #     except ValueError:
    #         evals = eigvals(T).real  # use fallback code but cast to real
    # else:
    evals = eigvals(T)  # nonreversible

    """Sort by decreasing absolute value"""
    ind = np.argsort(np.abs(evals))[::-1]
    evals = evals[ind]

    if isinstance(k, (list, set, tuple)):
        try:
            return [evals[n] for n in k]
        except IndexError:
            raise ValueError("given indices do not exist: ", k)
    elif k is not None:
        return evals[: k]
    else:
        return evals
