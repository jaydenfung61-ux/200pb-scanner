#!/usr/bin/env python3
"""
200 EMA Pullback Screener
Universe : NYSE + NASDAQ + TSX (full exchange listings)
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

_BAD_CHARS = set("$^+/~!")   # symbols containing these are warrants, notes, etc.

def _clean_us(symbols: pd.Series) -> list[str]:
    """Normalise US ticker symbols — drop non-equity suffixes, fix Berkshire-style spaces."""
    out = []
    for s in symbols.dropna().str.strip():
        if not s or any(c in s for c in _BAD_CHARS) or s.startswith("File"):
            continue
        out.append(s.replace(" ", "-"))   # e.g. "BRK A" → "BRK-A"
    return out


def get_nyse_tickers() -> list[str]:
    """All common-stock listings on NYSE from NASDAQ trader file."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
        headers=headers, verify=False, timeout=30,
    )
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    mask = (df["Exchange"] == "N") & (df["Test Issue"] == "N") & (df["ETF"] == "N")
    return _clean_us(df[mask]["ACT Symbol"])


def get_nasdaq_tickers() -> list[str]:
    """All common-stock listings on NASDAQ from NASDAQ trader file."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        headers=headers, verify=False, timeout=30,
    )
    df = pd.read_csv(io.StringIO(r.text), sep="|")
    mask = (df["Test Issue"] == "N") & (df["ETF"] == "N")
    return _clean_us(df[mask]["Symbol"])


def get_tsx_tickers() -> list[str]:
    """All listings on TSX via TMX company-directory JSON API."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(
        "https://www.tsx.com/json/company-directory/search/tsx/%5E*",
        headers=headers, verify=False, timeout=30,
    )
    results = r.json().get("results", [])
    tickers = []
    for item in results:
        symbol = item.get("symbol", "").strip()
        if symbol:
            tickers.append(f"{symbol.replace('.', '-')}.TO")
    return tickers


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
            "Exchange":      "TSX" if ticker.endswith(".TO") else "US (NYSE/NASDAQ)",
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
    nyse   = get_nyse_tickers()
    nasdaq = get_nasdaq_tickers()
    tsx    = get_tsx_tickers()
    tickers = list(dict.fromkeys(nyse + nasdaq + tsx))   # deduplicate, preserve order
    print(f"      {len(nyse)} NYSE  +  {len(nasdaq)} NASDAQ  +  {len(tsx)} TSX  =  {len(tickers)} total")

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
