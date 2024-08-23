from __future__ import annotations

import numpy as np
import CARCHINMean

from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from arch.univariate import EGARCH, ARCHInMean


def vecl(matrix):
    lower_matrix = np.tril(matrix, k=-1)
    array_with_zero = np.matrix(lower_matrix).A1
    array_without_zero = array_with_zero[array_with_zero != 0]

    return array_without_zero


def garch_to_u(res):
    # try:
    #     mu = res.params["mu"]
    #     kappa = res.params["kappa"]
    # except:
    #     mu = res.params["Const"]
    #     kappa = res.params["kappa"]
    cv = res.conditional_volatility
    # est_r = rets - mu - kappa * cv ** 2
    # USE res.std_resid
    std_res = res.std_resid  # est_r / h
    resid = res.resid
    udata = norm.cdf(std_res)
    return udata, cv, resid.values


def loglike_norm_dcc_copula(theta, udata, Dt_mat, resid_list):
    N, T = np.shape(udata)
    llf = np.zeros((T, 1))
    llf = llf
    trdata = np.array(norm.ppf(udata).T, ndmin=2)

    Rt, veclRt, Qt = dcceq(theta, trdata, Dt_mat, resid_list)

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

    Qbar = np.cov(eps)  # np.corrcoef(trdata.T)

    Qt = np.zeros((N, N, T))

    Qt[:, :, 0] = Qbar  # np.corrcoef(trdata.T)

    Rt = np.zeros((N, N, T))
    veclRt = np.zeros((T, int(N * (N - 1) / 2)))

    Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(Qt[:, :, 0])))
    Rt[:, :, 0] = Qstar_inv @ Qt[:, :, 0] @ Qstar_inv

    for j in range(1, T):
        Qt[:, :, j] = Qbar * (1 - a - b)
        Qt[:, :, j] = Qt[:, :, j] + a * np.outer(trdata[j - 1], trdata[j - 1])
        Qt[:, :, j] = Qt[:, :, j] + b * Qt[:, :, j - 1]
        Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(Qt[:, :, j])))
        Rt[:, :, j] = Qstar_inv @ Qt[:, :, j] @ Qstar_inv

    for j in range(0, T):
        veclRt[j, :] = vecl(Rt[:, :, j].T)
    return Rt, veclRt, Qt


def run_garch_on_return(rets, udata_list, model_parameters, Dt_mat, resid_list, arch_mean_type,
                        model="EGARCH_with_vol_in_mean"):
    i = 0
    for x in rets:
        tmp = rets[x].dropna()
        short_name = '_'.join(x.split())
        if model == "EGARCH_with_vol_in_mean":
            egim = CARCHINMean.CustomARCHInMean(tmp, form='var', volatility=EGARCH(p=1, o=1, q=1))
            result = egim.fit(arch_mean_type=arch_mean_type, update_freq=4, disp='off')
            model_parameters[short_name] = result

        udata, Dt, resid = garch_to_u(model_parameters[short_name])
        udata_list.append(udata)
        Dt_mat[i, i, :] = Dt
        resid_list[i, :] = resid
        i += 1

    return udata_list, Dt_mat, model_parameters


# DCC_EGARCH class
class DCC_EGARCH:

    def __init__(self, trdata, dccparams, a, b, sims, burn=1000):
        self.std_err = None
        self.volatility = None
        self.params = dccparams
        self.a = a
        self.b = b
        self.burn = burn
        self.sims = burn + sims
        self.N = dccparams.shape[0]
        self.trdata = trdata

    # Simulate univariate EGARCH(1,1) processes
    def simulate_egarch(self):  # n, omega, alpha, gamma, beta):
        volatility = np.empty([self.sims, self.N])  # matrix storing simulated volatilities
        std_err = np.empty([self.sims, self.N])  # matrix storing standard errors
        for asset in range(self.N):
            am = ARCHInMean(None, form='var', volatility=EGARCH(p=1, o=1, q=1))
            sim_data = am.simulate(
                [self.params[asset, 0], self.params[asset, 1], self.params[asset, 2], self.params[asset, 3],
                 self.params[asset, 4], self.params[asset, 5]], self.sims)
            volatility[:, asset] = sim_data['volatility']
            std_err[:, asset] = sim_data['errors'] / sim_data['volatility']

        self.volatility = volatility
        self.std_err = std_err

    def simulate_dcc2(self):
        std_err = np.empty([self.sims, self.N])  # matrix storing standard errors
        variance = np.empty([self.sims, self.N])  # matrix storing simulated volatilities
        voltry = np.empty([self.sims, self.N])
        R = np.zeros((self.N, self.N, self.sims))
        aQ = np.zeros((self.N, self.N, self.sims))
        # H = np.zeros((self.N, self.N))
        DCC_COVAR = np.zeros((self.N, self.N, self.sims))
        DCC_returns = np.zeros((self.sims, self.N))

        # Initialiser les matrices de corrélation et Q avec la corrélation moyenne
        mQ = np.corrcoef(self.trdata.T)
        aQ[:, :, 0] = mQ
        Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(aQ[:, :, 0])))
        R[:, :, 0] = Qstar_inv @ aQ[:, :, 0] @ Qstar_inv

        # simulate the zt for all t
        zt = multivariate_normal.rvs(mean=np.zeros(self.N), cov=np.eye(self.N), size=self.sims)
        for asset in range(self.N):
            mu, kappa, omega, alpha, gamma, beta = self.params[asset, :]
            variance[0, asset] = np.exp(omega / (1 - beta))

        D = np.diag(np.sqrt(variance[0, :]))
        H = D @ R[:, :, 0] @ D
        DCC_COVAR[:, :, 0] = H
        # Boucle principale pour la mise à jour de Q
        for t in range(1, self.sims):
            at_lag = cholesky(H) @ zt[t - 1, :]
            DCC_returns[t - 1, :] = self.params[:, 0] + self.params[:, 1] * np.diag(H) + at_lag
            std_err[t - 1, :] = np.diag(1.0 / np.diag(D)) @ at_lag
            for asset in range(self.N):
                mu, kappa, omega, alpha, gamma, beta = self.params[asset, :]
                variance[t, asset] = np.exp(
                    omega + alpha * (np.abs(zt[t - 1, asset]) - np.sqrt(2 / np.pi)) + gamma * zt[
                        t - 1, asset] + beta * np.log(variance[t - 1, asset]))

            aQ[:, :, t] = (mQ * (1 - self.a - self.b) + self.a * np.outer(std_err[t - 1], std_err[t - 1])
                           + self.b * aQ[:, :, t - 1])
            Qstar_inv = np.diag(1.0 / np.sqrt(np.diag(aQ[:, :, t])))
            R[:, :, t] = Qstar_inv @ aQ[:, :, t] @ Qstar_inv

            D = np.diag(np.sqrt(variance[t, :]))
            H = D @ R[:, :, t] @ D
            DCC_COVAR[:, :, t] = H
            if t == self.sims - 1:
                at = cholesky(H) @ zt[t, :]
                DCC_returns[t, :] = self.params[:, 0] + self.params[:, 1] * np.diag(H) + at
                std_err[t, :] = np.diag(1.0 / np.diag(D)) @ at
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
