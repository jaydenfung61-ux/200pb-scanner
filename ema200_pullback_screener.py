#!/usr/bin/env python3
"""
200 EMA Pullback Screener
Universe : S&P 500 + TSX 60
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

_SP500_TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","ABNB","AKAM","ALB","ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","APO","AAPL","AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB","AVY","AXON","BKR","BALL","BAC","BAX","BDX","BRK-B","BBY","TECH","BIIB","BLK","BX","BK","BA","BKNG","BSX","BMY","AVGO","BR","BRO","BF-B","BLDR","CHRW","CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE","CDW","CE","COR","CNC","CNX","CDAY","CF","CRL","SCHW","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS","CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","COP","ED","STZ","CEG","COO","CPRT","GLW","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS","DHR","DRI","DVA","DAY","DE","DAL","DVN","DXCM","FANG","DLR","DFS","DG","DLTR","D","DPZ","DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY","EG","EVRST","ES","EXC","EXPE","EXPD","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX","FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN","IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY","IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM","JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB","KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LLY","LIN","LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM","MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU","MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI","MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS","NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE","ORCL","OTIS","PCAR","PKG","PLTR","PH","PAYX","PAYC","PYPL","PNR","PEP","PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU","PEG","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN","RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB","STX","SRE","NOW","SHW","SPG","SWKS","SJM","SW","SNA","SOLV","SO","LUV","SWK","SBUX","STT","STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT","TEL","TDY","TFX","TER","TSLA","TXN","TPL","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB","TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO","VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRB","GWW","WAB","WBA","WMT","DIS","WBD","WM","WAT","WEC","WFC","WELL","WST","WDC","WHR","WRK","WY","WYNN","XEL","XYL","YUM","ZBRA","ZBH","ZTS",
]

_TSX60_TICKERS = [
    "ABX.TO","AEM.TO","AGF-B.TO","ALA.TO","AP-UN.TO","ARX.TO","ATD.TO","BAM.TO","BCE.TO","BN.TO",
    "BNS.TO","CAE.TO","CAR-UN.TO","CCO.TO","CCL-B.TO","CHP-UN.TO","CM.TO","CNQ.TO","CNR.TO","CP.TO",
    "CTC-A.TO","CVE.TO","DOL.TO","EMA.TO","ENB.TO","EQB.TO","FFH.TO","FM.TO","FNV.TO","FTS.TO",
    "GIB-A.TO","GWO.TO","H.TO","IFC.TO","IMO.TO","K.TO","KXS.TO","L.TO","LB.TO","LUN.TO",
    "MFC.TO","MG.TO","MRU.TO","NA.TO","NTR.TO","ONEX.TO","POU.TO","POW.TO","PPL.TO","RCI-B.TO",
    "RY.TO","SAP.TO","SLF.TO","SNC.TO","SU.TO","T.TO","TD.TO","TRP.TO","WCN.TO","WPM.TO",
]


def get_sp500_tickers() -> list[str]:
    return list(_SP500_TICKERS)


def get_tsx_tickers() -> list[str]:
    return list(_TSX60_TICKERS)


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
    sp500 = get_sp500_tickers()
    tsx60 = get_tsx_tickers()
    tickers = sp500 + tsx60
    print(f"      {len(sp500)} S&P 500  +  {len(tsx60)} TSX 60  =  {len(tickers)} total")

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

    fname = f"ema200_pullback_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(fname, index=True)
    print(f"\n  Saved → {fname}\n")


if __name__ == "__main__":
    main()
