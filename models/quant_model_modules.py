# models/quant_model_modules.py
import pandas as pd
import numpy as np
import pypfopt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import objective_functions

from scipy.optimize import minimize

# ---------- Helper ----------
def _weights_dict_to_array(wdict, columns):
    """clean_weights가 dict을 반환할 때 columns 순서에 맞춘 numpy array 반환"""
    return np.array([wdict.get(col, 0.0) for col in columns])

def clean_weights_array(arr, cutoff=1e-6, rounding=6):
    a = np.array(arr).copy()
    a[np.abs(a) < cutoff] = 0.0
    if rounding is not None:
        a = np.round(a, rounding)
    # renormalize small numerical drift
    s = a.sum()
    if s != 0:
        a = a / s
    return a

# ---------- Equal Weight ----------
def EW(returns:pd.DataFrame, rebal_periods:str)->pd.DataFrame:
    asset_num = len(returns.columns)
    rebal_indice = pd.date_range(returns.index[0], returns.index[-1], freq=rebal_periods) 
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]
    Pw = pd.DataFrame([[1/asset_num]*asset_num for i in range(len(rebal_indice))], index=rebal_indice, columns=returns.columns)
    return Pw

# ---------- Maximize Diversification (MD) ----------
def MaximizeDiversification(rebal_periods: str, returns: pd.DataFrame, lookback_periods:int, bnd=None, long_only=True, frequency=252):
    def calc_diversification_ratio(w, V):
        w = np.array(w)
        w_vol = np.dot(np.sqrt(np.diag(V)), w.T)
        port_vol = np.sqrt(np.dot(w.T,np.dot(V,w)))
        diversification_ratio = w_vol/port_vol if port_vol>0 else 0.0
        return -diversification_ratio

    def total_weight_constraint(x):
        return (x.sum() - 1.0)

    def get_weights(w0, V, bnd, long_only):
        cons = ({'type': 'eq', 'fun': total_weight_constraint},)
        if long_only and bnd is None:
            bnd = ([(0,1) for _ in range(len(w0))])
        res = minimize(calc_diversification_ratio, w0, bounds=bnd, args=(V,), method='SLSQP', constraints=cons)
        return clean_weights_array(res.x)

    w0 = np.array([1/len(returns.columns)]*len(returns.columns))
    rebal_indice = pd.date_range(returns.index[0+lookback_periods], returns.index[-1], freq=rebal_periods)
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]

    Pw_list = []
    for idx in rebal_indice:
        ext_returns = returns.loc[:idx].iloc[-lookback_periods:]
        V = pypfopt.risk_models.sample_cov(ext_returns, returns_data=True, frequency=frequency)
        W = get_weights(w0, V, bnd, long_only)
        Pw_list.append(W)
    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=returns.columns)
    return Pw

# ---------- Mean-Variance Max Sharpe ----------
def MeanVarianceMaxSharpe(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int,  frequency=252):
    rebal_indice = pd.date_range(returns.index[0+lookback_periods], returns.index[-1], freq=rebal_periods)
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]

    Pw_list = []
    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        mu = expected_returns.mean_historical_return(ext_df, returns_data=True, compounding=True, frequency=frequency)
        S = risk_models.sample_cov(ext_df, returns_data=True, frequency=frequency)

        ef = EfficientFrontier(mu, S)
        try:
            ef.max_sharpe()
            wdict = ef.clean_weights()
            W = _weights_dict_to_array(wdict, ext_df.columns)
            W = clean_weights_array(W)
        except Exception as e:
            # fallback to equal weight if optimization fails
            W = np.array([1/len(ext_df.columns)]*len(ext_df.columns))
        Pw_list.append(W)
    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=returns.columns)
    return Pw

# ---------- Mean-Variance Min Volatility (MVP) ----------
def MeanVarianceMinVolatility(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int, frequency=252):
    rebal_indice = pd.date_range(returns.index[0+lookback_periods], returns.index[-1], freq=rebal_periods)
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]

    Pw_list = []
    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        mu = expected_returns.mean_historical_return(ext_df, returns_data=True, compounding=True, frequency=frequency)
        S = risk_models.sample_cov(ext_df, returns_data=True, frequency=frequency)

        ef = EfficientFrontier(mu, S)
        try:
            ef.min_volatility()
            wdict = ef.clean_weights()
            W = _weights_dict_to_array(wdict, ext_df.columns)
            W = clean_weights_array(W)
        except Exception as e:
            W = np.array([1/len(ext_df.columns)]*len(ext_df.columns))
        Pw_list.append(W)
    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=returns.columns)
    return Pw

# ---------- Risk Parity (RP) ----------
def RP(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int, frequency=252, cov_type='simple'):
    from pypfopt.risk_models import exp_cov

    def weight_sum_constraint(x):
        return x.sum() - 1.0

    def weight_longonly(x):
        return x  # inequality constraint for >=0

    def get_covmat(rets, cov_type):
        if cov_type == "simple":
            return rets.cov().values
        elif cov_type == "exponential":
            return exp_cov(rets, returns_data=True, span=len(rets), frequency=frequency).values
        else:
            return rets.cov().values

    def risk_parity_objective(x, covmat):
        x = np.array(x)
        # portfolio volatility
        port_var = float(x.T @ covmat @ x)
        if port_var <= 0:
            return 1e9
        sigma = np.sqrt(port_var)
        mrc = (covmat @ x) / sigma
        rc = x * mrc
        # we want all rc equal -> minimize variance of rc
        return np.sum((rc - rc.mean())**2)

    rebal_indice = pd.date_range(returns.index[0+lookback_periods], returns.index[-1], freq=rebal_periods)
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]

    Pw_list = []
    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        covmat = get_covmat(ext_df, cov_type)

        x0 = np.repeat(1/covmat.shape[1], covmat.shape[1])
        cons = ({'type': 'eq', 'fun': weight_sum_constraint},
                {'type': 'ineq', 'fun': weight_longonly})
        res = minimize(fun=risk_parity_objective, x0=x0, args=(covmat,), method='SLSQP', constraints=cons, options={'maxiter':1000})
        if res.success:
            W = clean_weights_array(res.x)
        else:
            W = np.repeat(1/len(ext_df.columns), len(ext_df.columns))
        Pw_list.append(W)
    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=returns.columns)
    return Pw

# ---------- Risk Budget (RB) - optional ----------
def RB(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int, frequency=252, cov_type='simple', rb=None):
    from pypfopt.risk_models import exp_cov
    def obj_fun(x, p_cov, rb):
        return np.sum((x*np.dot(p_cov,x)/np.dot(x.transpose(), np.dot(p_cov, x))-rb)**2)

    def cons_sum_weight(x):
        return np.sum(x) - 1.0

    def cons_long_only_weight(x):
        return x

    def get_weights(asset_rets, rb_list, cov_type):
        num_arp = asset_rets.shape[1]
        if cov_type == "simple":
            p_cov = asset_rets.cov().values
        elif cov_type == "exponential":
            p_cov = exp_cov(asset_rets, returns_data=True, span=len(asset_rets), frequency=frequency).values
        w0 = 1.0 * np.ones((num_arp,)) / num_arp
        cons = ({'type': 'eq', 'fun':cons_sum_weight},
                {'type': 'ineq', 'fun':cons_long_only_weight})
        res = minimize(obj_fun, w0, args=(p_cov, rb_list), method='SLSQP', constraints=cons)
        return res.x

    if rb is None:
        rb = [1/len(returns.columns)]*len(returns.columns)

    rebal_indice = pd.date_range(returns.index[0+lookback_periods], returns.index[-1], freq=rebal_periods)
    rebal_indice = returns.index[returns.index.get_indexer(rebal_indice, method='ffill')]

    Pw_list = []
    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        W = get_weights(ext_df, rb, cov_type)
        Pw_list.append(W)

    Pw = pd.DataFrame(Pw_list, rebal_indice, columns=returns.columns)
    return Pw
