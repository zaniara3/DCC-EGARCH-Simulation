from __future__ import annotations

# import pickle
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint
import DCCEGARCH

# loading data
data = pd.read_csv('sectors_total_return_gross_dividend.csv', sep=',')

# convert data to monthly
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data.dropna()
ts_monthly = data.resample('M').last()

ts_monthly_returns = np.log(ts_monthly / ts_monthly.shift(1))
ts_monthly_returns = ts_monthly_returns.dropna()

sector_names = ts_monthly_returns.columns

# model
# params = {}
mean_vol = {sector: 0 for sector in sector_names}

scale = 1
ts_monthly_returns = scale * ts_monthly_returns
T, N = ts_monthly_returns.shape
model_parameters = {}
udata_list = []
Dt_mat = np.zeros((N, N, T))
resid_list = np.zeros((N, T))
arch_mean_type = 'standard'  # 'standard' #'half_var'
udata_list, Dt_mat, model_parameters = DCCEGARCH.run_garch_on_return(ts_monthly_returns, udata_list, model_parameters,
                                                                     Dt_mat, resid_list, arch_mean_type=arch_mean_type,
                                                                     model="EGARCH_with_vol_in_mean")
cons = ({'type': 'ineq', 'fun': lambda x: -x[0] - x[1] + 1})
lc = LinearConstraint([[1, 1]], -np.inf, 1)
bnds = ((0, 0.5), (0, 0.9997))

opt_out = minimize(DCCEGARCH.loglike_norm_dcc_copula, np.array([0.18, 0.8]), args=(udata_list, Dt_mat, resid_list),
                   bounds=bnds, constraints=cons, method='SLSQP')

# print(opt_out.success)
# print(opt_out.x)

# GARCH PARAMETERS
omega = [model_parameters[x].params['omega'] for x in model_parameters]
kappa = [model_parameters[x].params['kappa'] for x in model_parameters]
alpha = [model_parameters[x].params['alpha[1]'] for x in model_parameters]
beta = [model_parameters[x].params['beta[1]'] for x in model_parameters]
gamma = [model_parameters[x].params['gamma[1]'] for x in model_parameters]
muparam = [model_parameters[x].params['Const'] for x in model_parameters]

params = np.array([muparam, kappa, omega, alpha, gamma, beta]).transpose()  # parameters in a matrix form
a = opt_out.x[0]
b = opt_out.x[1]

trdata = np.array(norm.ppf(udata_list).T, ndmin=2)
Rt, veclRt, Qt, Qbar = DCCEGARCH.dcceq([a, b], trdata, Dt_mat, resid_list)

_, N, T = np.shape(Rt)
D_t = np.zeros((N, N, T))
i = 0
for x in ts_monthly_returns:
    short_name = '_'.join(x.split())
    tmp = model_parameters[short_name]
    D_t[i, i, :] = tmp.conditional_volatility
    i = i + 1

Ht = np.zeros((N, N, T))
for t in range(T):
    Ht[:, :, t] = D_t[:, :, t] @ Rt[:, :, t] @ D_t[:, :, t]

nsteps = 60
mIS = 100
mOOS = 100
np.random.seed(1245)

NRiskyAssets = len(omega)

# IS = in-sample to train the agent
simullogreturnsIS = np.empty([mIS, nsteps, NRiskyAssets])  # matrix storing simulated risky asset log-returns
simulvolatilitiesIS = np.empty([mIS, nsteps, NRiskyAssets])  # matrix storing simulated volatilities
simulcovarianceIS = np.empty([mIS, NRiskyAssets, NRiskyAssets, nsteps])  # matrix storing simulated covariances
simulcorrIS = np.empty([mIS, NRiskyAssets, NRiskyAssets, nsteps])  # matrix storing simulated covariances
simulQIS = np.empty([mIS, NRiskyAssets, NRiskyAssets, nsteps])  # matrix storing simulated Q matrix

# OS = out-of-sample, to test the agent performance on new data
simullogreturnsOOS = np.empty([mOOS, nsteps, NRiskyAssets])
simulvolatilitiesOOS = np.empty([mOOS, nsteps, NRiskyAssets])
simulcovarianceOOS = np.empty([mOOS, NRiskyAssets, NRiskyAssets, nsteps])
simulcorrOOS = np.empty([mOOS, NRiskyAssets, NRiskyAssets, nsteps])
simulQOOS = np.empty([mOOS, NRiskyAssets, NRiskyAssets, nsteps])

# Simulate log-returns and volatilities for the risky assets.
IS_sample = DCCEGARCH.DCC_EGARCH(trdata, params, a, b, sims=nsteps)
OS_sample = DCCEGARCH.DCC_EGARCH(trdata, params, a, b, sims=nsteps)

for i in range(mIS):
    IS_out = IS_sample.simulate_dcc2()
    simullogreturnsIS[i, :, :] = IS_out['DCC_returns']
    simulvolatilitiesIS[i, :, :] = IS_out['volatility']
    simulcovarianceIS[i, :, :, :] = IS_out['Ht']
    simulcorrIS[i, :, :, :] = IS_out['Rt']
    simulQIS[i, :, :, :] = IS_out['aQ']

for i in range(mOOS):
    OS_out = OS_sample.simulate_dcc2()
    simullogreturnsOOS[i, :, :] = OS_out['DCC_returns']
    simulvolatilitiesOOS[i, :, :] = OS_out['volatility']
    simulcovarianceOOS[i, :, :, :] = OS_out['Ht']
    simulcorrOOS[i, :, :, :] = OS_out['Rt']
    simulQOOS[i, :, :, :] = OS_out['aQ']

variable_list = {'simullogreturnsIS': simullogreturnsIS,
                 'simulvolatilitiesIS': simulvolatilitiesIS,
                 'simulcovarianceIS': simulcovarianceIS,
                 'simulcorrIS': simulcorrIS,
                 'simullogreturnsOOS': simullogreturnsOOS,
                 'simulvolatilitiesOOS': simulvolatilitiesOOS,
                 'simulcovarianceOOS': simulcovarianceOOS,
                 'simulcorrOOS': simulcorrOOS,
                 'sector_names': sector_names,
                 'NRiskyAssets': NRiskyAssets}

# with open('DCC_EGARCH_PATH.pkl', 'wb') as file:
#     pickle.dump(variable_list, file)

print('Done!')

pass
