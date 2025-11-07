# app.py
import io
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from models.quant_model_modules import (
    MeanVarianceMinVolatility,
    MeanVarianceMaxSharpe,
    RP,
    MaximizeDiversification,
    EW
)
from utils.portfolio_metrics import calc_portfolio_metrics

app = Flask(__name__, static_folder="static", template_folder="templates")

MODEL_MAP = {
    "MVP": MeanVarianceMinVolatility,
    "MaxSharpe": MeanVarianceMaxSharpe,
    "RP": RP,
    "MD": MaximizeDiversification,
    "EW": EW
}

@app.route("/")
def index():
    return render_template("index.html")

def _parse_csv(file_storage):
    # Expect CSV where first column is date (or index), other columns are asset returns (numeric)
    content = file_storage.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(content), index_col=0, parse_dates=True)
    return df

@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    """
    POST JSON:
    - model: 'MVP'|'MaxSharpe'|'RP'|'MD'|'EW'
    - rebal: pandas freq string e.g. 'W-FRI'
    - lookback: int
    - returns: optional (if client sends JSON data instead of file) dict of lists or CSV file via form 'file'
    """
    data = request.form.to_dict()
    json_payload = request.get_json(silent=True)  # allow JSON body as well

    # get model
    model_name = data.get("model") or (json_payload and json_payload.get("model"))
    if model_name is None:
        return jsonify({"error": "model required"}), 400
    if model_name not in MODEL_MAP:
        return jsonify({"error": f"unknown model {model_name}"}), 400

    rebal = data.get("rebal") or (json_payload and json_payload.get("rebal")) or "W-FRI"
    lookback = int(data.get("lookback") or (json_payload and json_payload.get("lookback")) or 60)

    # read returns data: prefer file upload
    if "file" in request.files and request.files["file"].filename != "":
        try:
            returns_df = _parse_csv(request.files["file"])
        except Exception as e:
            return jsonify({"error": f"failed parse csv: {str(e)}"}), 400
    else:
        # try JSON body 'returns'
        if json_payload and "returns" in json_payload:
            returns_df = pd.DataFrame(json_payload["returns"])
            # If index isn't datetime, attempt to keep as range index
            try:
                returns_df.index = pd.to_datetime(returns_df.index)
            except:
                pass
        else:
            return jsonify({"error": "no returns data provided"}), 400

    # ensure returns are numeric
    returns_df = returns_df.astype(float)

    # call model
    model_fn = MODEL_MAP[model_name]
    try:
        if model_name == "MD":
            Pw = model_fn(rebal, returns_df, lookback)
        elif model_name == "EW":
            Pw = model_fn(returns_df, rebal)
        else:
            Pw = model_fn(rebal, returns_df, lookback)
    except Exception as e:
        return jsonify({"error": f"model execution failed: {str(e)}"}), 500

    # use last weights for metrics
    last_weights = Pw.iloc[-1]
    metrics = calc_portfolio_metrics(returns_df, last_weights)

    resp = {
        "weights": last_weights.round(6).to_dict(),
        "metrics": metrics,
        "weights_over_time": Pw.round(6).to_dict(orient="index")
    }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True)
