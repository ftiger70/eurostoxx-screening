"""
Calcola gli indicatori tecnici dell'Eurostoxx 50 e li salva in un file JSON
(screening_data.json) che l'app iPhone legge via internet.
Pensato per essere eseguito da GitHub Actions, non manualmente sul PC
(per quello c'e' gia' aggiorna_screening.py che scrive direttamente nell'Excel).
"""

import json
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = "^STOXX50E"
OUT_FILE = "screening_data.json"


def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    log_ret = np.log(close / close.shift(1))
    hv30 = log_ret.rolling(30).std() * np.sqrt(252)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()

    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    min52 = low.rolling(252, min_periods=50).min()
    max52 = high.rolling(252, min_periods=50).max()

    supporto60 = low.rolling(60).min()
    resistenza60 = high.rolling(60).max()

    return dict(
        prezzo=close.iloc[-1], rsi=rsi.iloc[-1], hv30=hv30.iloc[-1],
        atr14=atr14.iloc[-1], ma50=ma50.iloc[-1], ma200=ma200.iloc[-1],
        min52=min52.iloc[-1], max52=max52.iloc[-1],
        supporto60=supporto60.iloc[-1], resistenza60=resistenza60.iloc[-1],
    )


def main():
    print(f"Scarico dati storici per {TICKER} ...")
    data = yf.download(TICKER, period="2y", interval="1d", progress=False, auto_adjust=False)
    if data.empty:
        raise SystemExit("Nessun dato scaricato da Yahoo Finance.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    ind = compute_indicators(data)
    for k, v in ind.items():
        if pd.isna(v):
            raise SystemExit(f"Dati insufficienti per calcolare '{k}'.")

    payload = {
        "updated_at": dt.datetime.utcnow().isoformat() + "Z",
        "ticker": "EUROSTOXX50 (SX5E)",
        "price": round(float(ind["prezzo"]), 2),
        "rsi14": round(float(ind["rsi"]), 1),
        "hv30_pct": round(float(ind["hv30"]) * 100, 2),
        "atr14": round(float(ind["atr14"]), 2),
        "ma50": round(float(ind["ma50"]), 2),
        "ma200": round(float(ind["ma200"]), 2),
        "min52": round(float(ind["min52"]), 2),
        "max52": round(float(ind["max52"]), 2),
        "support60": round(float(ind["supporto60"]), 2),
        "resistance60": round(float(ind["resistenza60"]), 2),
    }

    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print("Scritto", OUT_FILE)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
