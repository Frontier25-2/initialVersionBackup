
# utils/portfolio_metrics.py
# (가중치 → 기대수익률, 리스크, 샤프, 신뢰도 계산)
import numpy as np
import pandas as pd

def calc_portfolio_metrics(returns: pd.DataFrame, weights: pd.Series or np.ndarray, rf_rate=0.02, freq=252):
    """
    returns: daily returns DataFrame (columns = assets), index = date
    weights: pd.Series (index = columns) or numpy array aligned in same order
    rf_rate: annual risk-free rate (e.g., 0.02)
    freq: trading days in year (252) or calendar (365)
    """
    if isinstance(weights, pd.Series):
        w = weights.values
    else:
        w = np.array(weights)

    # annualized mean return and covariance
    mean_returns = returns.mean() * freq
    cov_matrix = returns.cov() * freq

    port_return = float(np.dot(w, mean_returns))
    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
    sharpe = float((port_return - rf_rate) / port_vol) if port_vol > 0 else 0.0

    # confidence: 간단한 정규화 (샤프 0~3 범위를 0~1로 매핑)
    confidence = min(max((sharpe / 3), 0), 1)

    return {
        "expected_return": round(port_return, 6),
        "risk": round(port_vol, 6),
        "sharpe_ratio": round(sharpe, 6),
        "confidence": round(confidence, 4)
    }
