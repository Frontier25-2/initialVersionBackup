# app.py (수정된 최종본)
import io
import base64
import matplotlib
matplotlib.use("Agg")  # 서버 환경에서 GUI 사용 금지
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# import only functions that exist in your quant_model_modules.py
from models.quant_model_modules import (
    MeanVarianceMaxSharpe,
    MeanVarianceMinVolatility,
    RP,
    MaximizeDiversification
)

# 한글 폰트 (Windows 환경)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)


def _parse_csv(file):
    """CSV 읽어서 (가격이면) 수익률로 변환 후 반환"""
    df = pd.read_csv(file, index_col=0, parse_dates=True)
    # heuristic: 가격이면 대부분 값이 5 이상 존재 -> 수익률로 변환
    try:
        if (df.abs() > 5).any().any():
            df = df.pct_change().dropna()
    except Exception:
        # 안전하게 수익률로 바꿀 수 없는 경우 그대로 반환
        pass
    return df


@app.route("/")
def index():
    # if you have index.html in templates, this will serve it
    from flask import render_template
    return render_template("index.html")


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    # validate file & params
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "CSV 파일을 업로드해주세요 (form field name='file')."}), 400

    file = request.files["file"]
    model = request.form.get("model") or request.json and request.json.get("model")
    rebal = request.form.get("rebal") or request.json and request.json.get("rebal") or "W-FRI"
    try:
        lookback = int(request.form.get("lookback") or (request.json and request.json.get("lookback")) or 60)
    except:
        return jsonify({"error": "lookback은 정수여야 합니다."}), 400

    returns_df = _parse_csv(file)
    # ensure numeric and sufficient data
    try:
        returns_df = returns_df.astype(float)
    except Exception as e:
        return jsonify({"error": f"returns data 변환 실패: {e}"}), 400

    # MODEL MAP: match the HTML select values you use
    MODEL_MAP = {
        "MVP": MeanVarianceMinVolatility,
        "MaxSharpe": MeanVarianceMaxSharpe,
        "RP": RP,
        "MD": MaximizeDiversification
    }

    if model not in MODEL_MAP:
        return jsonify({"error": f"unknown model '{model}'. 선택 가능한 값: {list(MODEL_MAP.keys())}"}), 400

    model_fn = MODEL_MAP[model]

    # call model function (note: model functions expect rebal_periods string)
    try:
        Pw = model_fn(rebal, returns_df, lookback)
    except Exception as e:
        return jsonify({"error": f"model execution failed: {str(e)}"}), 500

    # make sure Pw is a DataFrame and has proper columns
    if not isinstance(Pw, pd.DataFrame):
        return jsonify({"error": "model 반환값이 DataFrame이 아닙니다."}), 500

    # compute portfolio returns: use previous period weights (simple approximation)
    # Align weights and returns: assume Pw.index are rebalancing dates (DatetimeIndex or strings)
    # convert Pw index to datetime if possible for alignment
    try:
        Pw_index_dt = pd.to_datetime(Pw.index)
        Pw.index = Pw_index_dt
    except Exception:
        pass

    # create daily portfolio return series by forward-filling weights to daily dates
    try:
        # reindex Pw to returns_df index by forward filling weights
        Pw_daily = Pw.reindex(returns_df.index, method='ffill').fillna(method='ffill').fillna(0)
        port_ret = (Pw_daily.shift(1).fillna(0) * returns_df).sum(axis=1)
        cumret = (1 + port_ret).cumprod()
    except Exception as e:
        return jsonify({"error": f"포트폴리오 수익률 계산 실패: {e}"}), 500

    # convert indices to strings for JSON safety
    Pw_out = Pw.copy()
    Pw_out.index = Pw_out.index.astype(str)
    cumret_out = cumret.copy()
    cumret_out.index = cumret_out.index.astype(str)

    # plotting
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        cumret.plot(ax=ax, lw=1.5)
        ax.set_title(f"{model} 누적수익률")
        ax.set_ylabel("Cumulative return")
        ax.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
    except Exception as e:
        img_base64 = None
        print(f"[WARN] plot 생성 실패: {e}")

    resp = {
        "weights": Pw_out.round(6).to_dict(orient="index"),
        "cumret": cumret_out.to_dict(),
        "plot": img_base64
    }

    return jsonify(resp)


if __name__ == "__main__":
    app.run(debug=True)
