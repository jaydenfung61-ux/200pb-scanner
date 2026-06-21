#!/usr/bin/env python3
"""
200 EMA Pullback Screener
Universe : S&P 500 + S&P 400 + Nasdaq 100 + TSX 60 + TSX Composite
Strategy : Confirmed uptrend stocks pulling back to their 200-day EMA

Filters:
  1. Market gate  — SPY and QQQ above 50-day SMA
  2. ADV > 1M     — liquid names only
  3. Uptrend      — 50 EMA > 200 EMA, higher highs + higher lows over 12 months
  4. EMA proximity — price within -3% to +3% of 200-day EMA
  5. EMA slope    — 200-day EMA still sloping upward (not rolling over)
  6. Volume       — declining volume on pullback (no heavy distribution)
"""

import io
import warnings
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings("ignore")
requests.packages.urllib3.disable_warnings()

# ── Config ────────────────────────────────────────────────────────────────────

EMA_LOWER_PCT   = -3.0   # allow price up to 3% below 200 EMA
EMA_UPPER_PCT   =  3.0   # allow price up to 3% above 200 EMA
EMA_SLOPE_20D_MIN  = 0.5   # minimum % rise over 20 trading days
# Multi-timeframe lookbacks (trading days)
EMA_SLOPE_1M   =  21   # ~1 month
EMA_SLOPE_3M   =  63   # ~3 months
EMA_SLOPE_6M   = 126   # ~6 months
VOL_RECENT_DAYS =  15    # recent window for declining-volume check
VOL_BASE_DAYS   =  35    # baseline window (prior to recent)
HH_HL_HALF      = 126    # half of 12-month window (~6 months each half)
MIN_ADV         = 1_000_000


# ── Universe ──────────────────────────────────────────────────────────────────

def get_sp500_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers=headers, verify=False, timeout=15,
    )
    df = pd.read_html(io.StringIO(r.text))[0]
    return df["Symbol"].str.replace(".", "-", regex=False).tolist()


def get_tsx_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        headers=headers, verify=False, timeout=15,
    )
    tables = pd.read_html(io.StringIO(r.text))
    df = next(t for t in tables if "Symbol" in t.columns and len(t) >= 50)
    tickers = df["Symbol"].dropna().str.strip().tolist()
    return [
        t if t.endswith(".TO") else f"{t.replace('.', '-')}.TO"
        for t in tickers
    ]


def get_sp400_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        headers=headers, verify=False, timeout=15,
    )
    df = pd.read_html(io.StringIO(r.text))[0]
    col = next(c for c in df.columns if c in ("Ticker", "Symbol", "Ticker symbol"))
    return df[col].str.replace(".", "-", regex=False).tolist()


def get_nasdaq100_tickers() -> list[str]:
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        headers=headers, verify=False, timeout=15,
    )
    tables = pd.read_html(io.StringIO(r.text))
    for t in tables:
        col = next((c for c in t.columns if c in ("Ticker", "Symbol")), None)
        if col and len(t) >= 90:
            return t[col].str.replace(".", "-", regex=False).tolist()
    return []


def get_tsx_composite_tickers() -> list[str]:
    """S&P/TSX Composite (~220 stocks) via Wikipedia constituent table."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index",
        headers=headers, verify=False, timeout=15,
    )
    tables = pd.read_html(io.StringIO(r.text))
    df = next(t for t in tables if "Ticker" in t.columns and len(t) >= 200)
    tickers = df["Ticker"].dropna().str.strip().tolist()
    return [f"{t}.TO" if not t.endswith(".TO") else t for t in tickers]


# ── Market Gate ───────────────────────────────────────────────────────────────

def check_market_gate() -> tuple[bool, str]:
    proxies = ["SPY", "QQQ", "XIU.TO"]
    data = yf.download(proxies, period="3mo", progress=False, auto_adjust=True)
    failed = []
    for ticker in proxies:
        close = data["Close"][ticker].dropna()
        sma50 = close.rolling(50).mean().iloc[-1]
        price = close.iloc[-1]
        if float(price) < float(sma50):
            failed.append(ticker)
    if failed:
        return False, f"Gate failed: {', '.join(failed)} below 50-day SMA"
    return True, "SPY, QQQ, and XIU.TO all above 50-day SMA"


# ── Screening Filters ─────────────────────────────────────────────────────────

def is_uptrending(close: pd.Series, ema50: pd.Series, ema200: pd.Series) -> bool:
    """
    Three-part uptrend check:
      1. 50 EMA above 200 EMA  (trend structure aligned)
      2. Higher high in recent 6 months vs prior 6 months
      3. Higher low in recent 6 months vs prior 6 months
    """
    if len(close) < HH_HL_HALF * 2:
        return False

    if float(ema50.iloc[-1]) <= float(ema200.iloc[-1]):
        return False

    prior  = close.iloc[-(HH_HL_HALF * 2):-HH_HL_HALF]
    recent = close.iloc[-HH_HL_HALF:]

    hh = float(recent.max()) > float(prior.max())
    hl = float(recent.min()) > float(prior.min())
    return hh and hl


def ema200_sloping_up(ema200: pd.Series) -> bool:
    """
    Two-part slope rule:
      1. Short-term momentum: EMA rose > 0.5% over the last 20 trading days
      2. Multi-timeframe: EMA today > EMA 1 month ago, 3 months ago, AND 6 months ago
    All conditions must be true simultaneously.
    """
    if len(ema200) < EMA_SLOPE_6M + 1:
        return False
    now  = float(ema200.iloc[-1])
    d20  = float(ema200.iloc[-20])
    m1   = float(ema200.iloc[-EMA_SLOPE_1M])
    m3   = float(ema200.iloc[-EMA_SLOPE_3M])
    m6   = float(ema200.iloc[-EMA_SLOPE_6M])
    short_term  = (now - d20) / d20 * 100 > EMA_SLOPE_20D_MIN
    multi_frame = now > m1 and now > m3 and now > m6
    return short_term and multi_frame


def price_near_ema200(price: float, ema200_val: float) -> tuple[bool, float]:
    pct = (price - ema200_val) / ema200_val * 100
    return EMA_LOWER_PCT <= pct <= EMA_UPPER_PCT, round(pct, 2)


def volume_declining(volume: pd.Series) -> bool:
    """Recent avg volume lower than baseline — low-distribution pullback."""
    needed = VOL_RECENT_DAYS + VOL_BASE_DAYS
    if len(volume) < needed:
        return False
    recent   = float(volume.iloc[-VOL_RECENT_DAYS:].mean())
    baseline = float(volume.iloc[-(needed):-VOL_RECENT_DAYS].mean())
    return recent < baseline


# ── Per-ticker Screen ─────────────────────────────────────────────────────────

def screen_ticker(ticker: str, data) -> dict | None:
    try:
        close = data["Close"][ticker].dropna()
        vol   = data["Volume"][ticker].dropna()

        if len(close) < HH_HL_HALF * 2 + 10:
            return None

        price = float(close.iloc[-1])
        adv   = float(vol.iloc[-20:].mean())

        if adv < MIN_ADV:
            return None

        ema50  = close.ewm(span=50,  adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()

        # Uptrend: 50 EMA > 200 EMA + HH/HL over 12 months
        if not is_uptrending(close, ema50, ema200):
            return None

        # 200 EMA must still be sloping upward
        if not ema200_sloping_up(ema200):
            return None

        # Price must be within -3% to +3% of 200 EMA
        ema200_val = float(ema200.iloc[-1])
        near, pct_from_ema = price_near_ema200(price, ema200_val)
        if not near:
            return None

        vol_ok = volume_declining(vol)

        return {
            "Ticker":        ticker,
            "Exchange":      "TSX" if ticker.endswith(".TO") else "US",
            "Price":         round(price, 2),
            "EMA200":        round(ema200_val, 2),
            "% from EMA":    pct_from_ema,
            "EMA50":         round(float(ema50.iloc[-1]), 2),
            "Vol Declining": "✓" if vol_ok else "—",
            "ADV (M)":       round(adv / 1e6, 1),
        }
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    bar = "━" * 56
    print(f"\n{bar}")
    print(f"  200 EMA Pullback Screener  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(bar)

    # 1 — Market gate
    print("\n[1/4] Market Gate")
    ok, msg = check_market_gate()
    print(f"      {msg}")
    if not ok:
        print("\n  Unfavorable market conditions — aborting.\n")
        return

    # 2 — Universe
    print("\n[2/4] Fetching universe")
    sp500     = get_sp500_tickers()
    sp400     = get_sp400_tickers()
    ndx100    = get_nasdaq100_tickers()
    tsx60     = get_tsx_tickers()
    tsx_comp  = get_tsx_composite_tickers()
    seen = set()
    tickers = []
    for t in sp500 + sp400 + ndx100 + tsx60 + tsx_comp:
        if t not in seen:
            seen.add(t)
            tickers.append(t)
    print(
        f"      {len(sp500)} S&P 500  +  {len(sp400)} S&P 400  +  "
        f"{len(ndx100)} Nasdaq 100  +  {len(tsx60)} TSX 60  +  "
        f"{len(tsx_comp)} TSX Composite  =  {len(tickers)} unique"
    )

    # 3 — Download 14 months (need 252 days for HH/HL + 200-day EMA warmup)
    print("\n[3/4] Downloading 14 months of price/volume data…")
    data = yf.download(tickers, period="14mo", progress=False, auto_adjust=True)
    print(f"      Data loaded")

    # 4 — Screen
    print("\n[4/4] Applying filters…")
    results = [r for t in tickers if (r := screen_ticker(t, data)) is not None]

    if not results:
        print("      No stocks passed all filters today.\n")
        return

    df = (
        pd.DataFrame(results)
        .sort_values("% from EMA")
        .reset_index(drop=True)
    )
    df.index += 1

    print(f"\n{bar}")
    print(f"  PASSED: {len(df)} stocks")
    print(f"  Sorted: closest to 200 EMA first (best pullback entries at top)")
    print(bar)
    print()
    print(df.to_string())

    fname = f"/Users/jaydenfung/ema200_pullback_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=True)
    print(f"\n  Saved → {fname}\n")


if __name__ == "__main__":
    main()
