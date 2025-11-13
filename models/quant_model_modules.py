# >>> paste into models/quant_model_modules.py (replace existing implementations)
import pandas as pd
import numpy as np
import pypfopt

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from scipy.optimize import minimize

# ---------- Helper ----------
def _weights_dict_to_array(wdict, columns):
    return np.array([wdict.get(col, 0.0) for col in columns])

def clean_weights_array(arr, cutoff=1e-8, rounding=6):
    a = np.array(arr).copy()
    a[np.abs(a) < cutoff] = 0.0
    if rounding is not None:
        a = np.round(a, rounding)
    s = a.sum()
    if s != 0:
        a = a / s
    else:
        # all zeros -> equal weight
        a = np.ones_like(a) / len(a)
    return a

def _clean_returns_df(returns_df):
    # remove inf and drop rows with any NaN
    df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return df

# ---------- utility: build rebalancing indices ----------
def _build_rebal_index(returns_index: pd.DatetimeIndex, lookback_periods: int, rebal_periods: str):
    """
    returns_index: original returns index (DatetimeIndex)
    lookback_periods: int
    rebal_periods: pandas offset alias like 'W-FRI'
    """
    # start from returns_index[0 + lookback_periods] to ensure lookback exists
    if len(returns_index) <= lookback_periods:
        # not enough data, fallback to full index
        rebal_indice = returns_index
    else:
        start = returns_index[0 + lookback_periods]
        end = returns_index[-1]
        # create candidate dates then align to nearest available index with ffill
        cand = pd.date_range(start, end, freq=rebal_periods)
        # map to actual dates in returns_index (ffill)
        rebal_indice = returns_index[returns_index.get_indexer(cand, method='ffill')]
    # ensure unique and sorted
    rebal_indice = rebal_indice.unique()
    rebal_indice = rebal_indice[rebal_indice.notnull()]
    return rebal_indice

# ---------- Maximize Diversification (MD) ----------
def MaximizeDiversification(rebal_periods: str, returns: pd.DataFrame, lookback_periods:int, bnd=None, long_only=True, frequency=252):
    returns = _clean_returns_df(returns)
    cols = returns.columns

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
        try:
            res = minimize(calc_diversification_ratio, w0, bounds=bnd, args=(V,), method='SLSQP', constraints=cons)
            W = clean_weights_array(res.x)
        except Exception as e:
            # fallback equal weight
            W = np.ones(len(w0)) / len(w0)
        return W

    w0 = np.array([1/len(cols)]*len(cols))
    rebal_indice = _build_rebal_index(returns.index, lookback_periods, rebal_periods)

    Pw_list = []
    for idx in rebal_indice:
        ext_returns = returns.loc[:idx].iloc[-lookback_periods:]
        # robust covariance via pypfopt
        try:
            V = pypfopt.risk_models.sample_cov(ext_returns, returns_data=True, frequency=frequency)
        except Exception:
            V = ext_returns.cov().values
        W = get_weights(w0, V, bnd, long_only)
        Pw_list.append(W)

    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=cols)
    return Pw

# ---------- Mean-Variance Max Sharpe ----------
def MeanVarianceMaxSharpe(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int,  frequency=252):
    returns = _clean_returns_df(returns)
    cols = returns.columns

    rebal_indice = _build_rebal_index(returns.index, lookback_periods, rebal_periods)
    Pw_list = []

    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        # if ext_df is too small, fallback immediately
        if ext_df.shape[0] < max(10, len(cols)+2):
            W = np.ones(len(cols)) / len(cols)
            Pw_list.append(W)
            continue

        try:
            mu = expected_returns.mean_historical_return(ext_df, returns_data=True, compounding=True, frequency=frequency)
            S = risk_models.sample_cov(ext_df, returns_data=True, frequency=frequency)
            ef = EfficientFrontier(mu, S)
            ef.max_sharpe()
            wdict = ef.clean_weights()
            W = _weights_dict_to_array(wdict, ext_df.columns)
            W = clean_weights_array(W)
        except Exception as e:
            # log and fallback
            print(f"[WARN] MeanVarianceMaxSharpe optimization failed at {idx}: {e}")
            W = np.ones(len(cols)) / len(cols)

        Pw_list.append(W)

    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=cols)
    return Pw

# ---------- Mean-Variance Min Volatility (MVP) ----------
def MeanVarianceMinVolatility(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int, frequency=252):
    returns = _clean_returns_df(returns)
    cols = returns.columns

    rebal_indice = _build_rebal_index(returns.index, lookback_periods, rebal_periods)
    Pw_list = []

    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        if ext_df.shape[0] < max(10, len(cols)+2):
            W = np.ones(len(cols)) / len(cols)
            Pw_list.append(W)
            continue

        try:
            mu = expected_returns.mean_historical_return(ext_df, returns_data=True, compounding=True, frequency=frequency)
            S = risk_models.sample_cov(ext_df, returns_data=True, frequency=frequency)
            ef = EfficientFrontier(mu, S)
            ef.min_volatility()
            wdict = ef.clean_weights()
            W = _weights_dict_to_array(wdict, ext_df.columns)
            W = clean_weights_array(W)
        except Exception as e:
            print(f"[WARN] MeanVarianceMinVolatility failed at {idx}: {e}")
            W = np.ones(len(cols)) / len(cols)

        Pw_list.append(W)

    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=cols)
    return Pw

# ---------- Risk Parity (RP) ----------
def RP(rebal_periods:str, returns:pd.DataFrame, lookback_periods:int, frequency=252, cov_type='simple'):
    from pypfopt.risk_models import exp_cov
    returns = _clean_returns_df(returns)
    cols = returns.columns

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
        port_var = float(x.T @ covmat @ x)
        if port_var <= 0:
            return 1e9
        sigma = np.sqrt(port_var)
        mrc = (covmat @ x) / sigma
        rc = x * mrc
        return np.sum((rc - rc.mean())**2)

    rebal_indice = _build_rebal_index(returns.index, lookback_periods, rebal_periods)
    Pw_list = []

    for idx in rebal_indice:
        ext_df = returns.loc[:idx].iloc[-lookback_periods:]
        if ext_df.shape[0] < max(10, len(cols)+2):
            W = np.ones(len(cols)) / len(cols)
            Pw_list.append(W)
            continue

        covmat = get_covmat(ext_df, cov_type)

        x0 = np.repeat(1/covmat.shape[1], covmat.shape[1])
        cons = ({'type': 'eq', 'fun': weight_sum_constraint},
                {'type': 'ineq', 'fun': weight_longonly})
        try:
            res = minimize(fun=risk_parity_objective, x0=x0, args=(covmat,), method='SLSQP', constraints=cons, options={'maxiter':1000})
            if res.success:
                W = clean_weights_array(res.x)
            else:
                W = np.ones(len(cols)) / len(cols)
        except Exception as e:
            print(f"[WARN] RP failed at {idx}: {e}")
            W = np.ones(len(cols)) / len(cols)

        Pw_list.append(W)

    Pw = pd.DataFrame(Pw_list, index=rebal_indice, columns=cols)
    return Pw
