from __future__ import annotations

import numpy as np
import CARCHINMean

from scipy.stats import norm
from scipy.stats import multivariate_normal
from arch.univariate import EGARCH


def vecl(matrix):
    lower_matrix = np.tril(matrix, k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero != 0]

    return array_without_zero


def garch_to_u(res):
    cv = res.conditional_volatility
    std_res = res.std_resid
    resid = res.resid
    udata = norm.cdf(std_res)
    return udata, cv, resid.values


def loglike_norm_dcc_copula(theta, udata, Dt_mat, resid_list):
    N, T = np.shape(udata)
    llf = np.zeros((T, 1))
    llf = llf
    trdata = np.array(norm.ppf(udata).T, ndmin=2)

    Rt, veclRt, Qt, Qbar = dcceq(theta, trdata, Dt_mat, resid_list)

    for i in range(0, T):
        llf[i] = -0.5 * np.log(np.linalg.det(Rt[:, :, i]))
        llf[i] = llf[i] - 0.5 * np.matmul(np.matmul(trdata[i, :], (np.linalg.inv(Rt[:, :, i]) - np.eye(N))),
                                          trdata[i, :].T)
    llf = np.sum(llf)

    return -llf


def dcceq(theta, trdata, Dt_mat, resid_list):
    T, N = np.shape(trdata)
    a, b = theta
    if min(a, b) < 0 or max(a, b) > 1 or a + b > .999999:
        a = .9999 - b
    eps = np.zeros((N, T))
    for i in range(T):
        eps[:, i] = np.matmul(np.diag(1 / np.diag(Dt_mat[:, :, i])), resid_list[:, i])

    Qbar = np.cov(eps)

    Qt = np.zeros((N, N, T))

    Qt[:, :, 0] = Qbar

    Rt = np.zeros((N, N, T))
    veclRt = np.zeros((T, int(N * (N - 1) / 2)))

    Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(Qt[:, :, 0])))
    Rt[:, :, 0] = Qstar_inv @ Qt[:, :, 0] @ Qstar_inv

    for j in range(1, T):
        Qt[:, :, j] = Qbar * (1 - a - b)
        Qt[:, :, j] = Qt[:, :, j] + a * np.outer(eps[:, j - 1], eps[:, j - 1])
        Qt[:, :, j] = Qt[:, :, j] + b * Qt[:, :, j - 1]
        Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(Qt[:, :, j])))
        Rt[:, :, j] = Qstar_inv @ Qt[:, :, j] @ Qstar_inv

    for j in range(0, T):
        veclRt[j, :] = vecl(Rt[:, :, j].T)
    return Rt, veclRt, Qt, Qbar


def run_garch_on_return(rets, udata_list, model_parameters, Dt_mat, resid_list, arch_mean_type,
                        model="EGARCH_with_vol_in_mean"):
    i = 0
    short_name = None
    for x in rets:
        tmp = rets[x].dropna()
        if model == "EGARCH_with_vol_in_mean":
            egim = CARCHINMean.CustomARCHInMean(tmp, form='var', volatility=EGARCH(p=1, o=1, q=1))
            result = egim.fit(arch_mean_type=arch_mean_type, update_freq=4, disp='off')
            short_name = '_'.join(x.split())
            model_parameters[short_name] = result

        udata, Dt, resid = garch_to_u(model_parameters[short_name])
        udata_list.append(udata)
        Dt_mat[i, i, :] = Dt
        resid_list[i, :] = resid
        i += 1

    return udata_list, Dt_mat, model_parameters


# DCC_EGARCH class
class DCC_EGARCH:

    def __init__(self, trdata, dccparams, a, b, sims, nburn=2000):
        self.params = dccparams
        self.a = a
        self.b = b
        self.burn = nburn
        self.sims = nburn + sims
        self.N = dccparams.shape[0]
        self.trdata = trdata
        self.volatility = None
        self.unierrors = None

    # Simulate univariate EGARCH(1,1) processes
    def simulate_egarch(self):
        variance_uni = np.empty([self.sims, self.N])
        uniret = np.empty([self.sims, self.N])
        unierrors = np.empty([self.sims, self.N])
        variance_uni[0, :] = np.exp(self.params[:, 2] / (1 - self.params[:, 5]))
        zt = multivariate_normal.rvs(mean=np.zeros(self.N), cov=np.eye(self.N), size=self.sims)
        unierrors[0, :] = np.sqrt(variance_uni[0, :]) * zt[0, :]
        uniret[0, :] = self.params[:, 0] + self.params[:, 1] * variance_uni[0, :]

        for t in range(1, self.sims):
            variance_uni[t, :] = np.exp(
                self.params[:, 2] + self.params[:, 3] * (np.abs(zt[t - 1, :]) - np.sqrt(2 / np.pi))
                + self.params[:, 4] * zt[t - 1, :] + self.params[:, 5] * np.log(variance_uni[t - 1, :]))
            unierrors[t, :] = np.sqrt(variance_uni[t, :]) * zt[t, :]
            uniret[t, :] = self.params[:, 0] + self.params[:, 1] * variance_uni[t, :] + unierrors[t, :]
        self.volatility = np.sqrt(variance_uni)
        self.unierrors = unierrors

    def simulate_dcc2(self):
        std_err = np.empty([self.sims, self.N])  # matrix storing standard errors
        voltry = np.empty([self.sims, self.N])
        R = np.zeros((self.N, self.N, self.sims))
        aQ = np.zeros((self.N, self.N, self.sims))
        DCC_COVAR = np.zeros((self.N, self.N, self.sims))
        DCC_returns = np.zeros((self.sims, self.N))

        self.simulate_egarch()
        variance = self.volatility ** 2
        mQ = np.cov(self.trdata.T)
        aQ[:, :, 0] = mQ
        Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(aQ[:, :, 0])))
        R[:, :, 0] = Qstar_inv @ aQ[:, :, 0] @ Qstar_inv
        D = np.diag(np.sqrt(variance[0, :]))
        H = D @ R[:, :, 0] @ D
        DCC_COVAR[:, :, 0] = H
        at_lag = multivariate_normal.rvs(mean=np.zeros(self.N), cov=DCC_COVAR[:, :, 0])
        DCC_returns[0, :] = self.params[:, 0] + self.params[:, 1] * np.diag(H) + at_lag
        # Boucle principale pour la mise à jour de Q
        for t in range(1, self.sims):
            std_err[t - 1, :] = np.diag(1.0 / np.diag(D)) @ at_lag
            aQ[:, :, t] = mQ * (1 - self.a - self.b) + self.a * np.outer(std_err[t - 1, :],
                                                                         std_err[t - 1, :]) + self.b * aQ[:, :, t - 1]
            Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(aQ[:, :, t])))
            R[:, :, t] = Qstar_inv @ aQ[:, :, t] @ Qstar_inv

            D = np.diag(np.sqrt(variance[t, :]))
            H = D @ R[:, :, t] @ D
            DCC_COVAR[:, :, t] = H

            at_lag = multivariate_normal.rvs(mean=np.zeros(self.N), cov=DCC_COVAR[:, :, t])
            DCC_returns[t, :] = self.params[:, 0] + self.params[:, 1] * np.diag(H) + at_lag
        out = {
            "Rt": R[:, :, self.burn:],  # Corrélations dynamiques
            "volatility": np.sqrt(variance[self.burn:, :]),  # Volatilités univariées
            "voltry": voltry[self.burn:, :],  # Volatilités univariées
            "Ht": DCC_COVAR[:, :, self.burn:],  # Matrices de covariance
            "DCC_returns": DCC_returns[self.burn:, :],  # Rendements simulés
            "mQ": mQ,
            "aQ": aQ[:, :, self.burn:]
        }
        return out
