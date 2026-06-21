#!/usr/bin/env python3
"""
200 EMA Pullback Screener
Universe : NYSE
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

# NYSE-listed common stocks (liquid names across all sectors)
_NYSE_TICKERS = [
    # Financials
    "JPM","BAC","WFC","C","GS","MS","AXP","BLK","BX","KKR","APO","AIG","MET","PRU","AFL","ALL",
    "TRV","CB","HIG","L","GL","LNC","UNM","PFG","CINF","AIZ","BK","STT","NTRS","RF","KEY","CFG",
    "MTB","USB","FHN","HBAN","SNV","CMA","FNB","BOH","GBCI","IBOC","NYCB","OFG","PACW","BANC",
    "FHB","CVBF","WAFD","WSFS","FFIN","BOKF","UMBF","WBS","FULT","PBCT","TCF","CBSH","ONB",
    "TRMK","SFNC","BUSE","HOPE","HAFC","BANR","HTLF","QCRH","FMBH","NBHC","CATC","SBCF",
    "ICE","CME","CBOE","NDAQ","MKTX","VIRT","LAZ","EVR","HLI","MC","PJT","GHL","PIPR",
    "IVZ","AMG","WDR","VRTS","CNS","APAM","NOAH","VCNX",
    # Technology (NYSE-listed)
    "IBM","HPE","HPQ","DELL","NCR","JNPR","GLW","TEL","APH","CDW","LDOS","SAIC","CSRA",
    "DXC","ACN","ADP","PAYX","FIS","FISV","FLT","WEX","EVTC","WU","MDP","XRX",
    # Healthcare
    "JNJ","PFE","MRK","ABT","BMY","LLY","MDT","BDX","BAX","EW","SYK","BSX","ZBH","HAE",
    "XRAY","HOLX","VAR","NVCR","TFX","OMI","PKI","DHR","TMO","WAT","A","IQV","IQVIA",
    "HCA","THC","UHS","CYH","ENSG","SEM","ADUS","AMED","LH","DGX","EVHC","PDCO","HSIC",
    "MCK","CAH","ABC","CVS","WBA","ESRX","ANTM","UNH","CI","HUM","CNC","MOH","WCG","HQY",
    # Energy
    "XOM","CVX","COP","EOG","PXD","DVN","MRO","APA","HES","VLO","PSX","MPC","DK","PBF",
    "PARR","CLMT","SUN","CVR","DINO","HollyFrontier","OXY","OVV","FANG","CLR","WPX","CPE",
    "CXO","JAX","SM","REI","ESTE","MTDR","MGY","RRC","EQT","CNX","AR","SWN","COG","CHK",
    "WTI","NOG","CIVI","PDCE","BATL","KOS","VNOM","BSM",
    "SLB","HAL","BKR","NOV","FTI","WHD","LBRT","PTEN","HP","NE","DO","VAL","RIG",
    "OKE","WMB","KMI","ET","EPD","MMP","PAA","TRGP","DT","ENLC","CEQP","SMLP",
    "SO","DUK","NEE","AEP","EXC","D","ED","FE","PCG","EIX","PEG","PPL","CMS","NI",
    "AES","ETR","WEC","DTE","LNT","EVRG","AEE","PNW","XEL","IDA","AVA","NWE","POR",
    # Consumer Discretionary
    "AMZN","HD","LOW","TGT","WMT","COST","TJX","ROST","M","KSS","JWN","DDS","SKT",
    "SPG","MAC","CBL","WPG","PEI","TCO","BRX","KIM","REG","SITC","WRI","EQY","ROIC",
    "MCD","YUM","QSR","DRI","TXRH","CAKE","DPZ","PZZA","JACK","WEN","CMG","SBUX",
    "NKE","VFC","PVH","RL","TPR","HBI","G","COH","KOR","LEVI","UAA","UA","SKX","OXM",
    "GM","F","TM","HMC","RACE","PAG","KMX","AN","LAD","ABG","GPI","SAH","CVNA","CAR","HTZ",
    "CCL","RCL","NCLH","MHO","MDC","PHM","LEN","DHI","TOL","NVR","BZH","KBH","MHO","HOV",
    "LKQ","AAP","AZO","ORLY","BWA","LEA","MGA","ALV","TEN","WNC","XHB",
    # Consumer Staples
    "PG","KO","PEP","PM","MO","BTI","MDLZ","GIS","K","CPB","CAG","SJM","HRL","MKC",
    "CL","CHD","CLX","EL","REV","AVP","COTY","IFF","SYY","US","USFD","PFGC","CHEF",
    "KR","SFM","WMT","TGT","ACI","SVU","VLGEA","WEIS","INGR","FDP","BG","ADM","CALM",
    "TSN","HRL","PPC","SAFM","MFI","SEB","LANC","JJSF","THS","SMPL","POST","TWNK",
    # Industrials
    "HON","GE","MMM","RTX","LMT","BA","NOC","GD","L3H","TXT","HEI","HEICO","TDG",
    "SPR","KTOS","AJRD","CW","MOOG","DRS","HXL","TDY","FLIR","ESLT","LDOS","SAIC",
    "CAT","DE","CMI","PCAR","NAV","ALSN","TRN","GBX","WAB","TT","ITW","EMR","ROK",
    "ROP","CARR","OTIS","JCI","AAON","WTS","AOS","ACCO","MWA","NVT","REXL",
    "UPS","FDX","XPO","SAIA","ODFL","JBHT","KNX","CHRW","EXPD","FWRD","ECHO",
    "GWW","MSC","FAST","SNA","KMT","TKR","ATI","ARNC","CMC","NUE","STLD","X","CLF",
    "AME","PH","DANAHER","IEX","TRMB","FTV","GNSS","RBC","NDSN","EFX","EXPO","MASI",
    "WM","RSG","WCN","CVA","SRCL","CARG","CLH","US","HDSN",
    # Materials
    "LIN","APD","PPG","SHW","ECL","EMN","LYB","HUN","CC","TROX","VNTR","ASIX","KRA",
    "NEM","AEM","ABX","AU","AUY","GFI","KGC","EGO","IAG","AGI","OR","WPM","FNV","SA",
    "FCX","SCCO","TCK","HBM","CS","TECK","ACH","CENX","AA","KALU","CRS","ATI","ARNC",
    "IP","PKG","SEE","SON","SLGN","BERY","PTVE","MERC","CLW","GAPFF","UFI","FUL","IFF",
    "MLM","VMC","SUM","EXP","MDU","USG","USCR","SLCA","BOOM","STRL","GVP",
    # Real Estate
    "AMT","CCI","EQIX","SBAC","DLR","QTS","COR","IRM","CONE","NSA","LSI","CUBE",
    "EXR","PSA","LIFE","SSS","SP","JELD","REXNORD",
    "EQR","AVB","MAA","UDR","CPT","AIV","ACC","EDR","IRT","NXRT","APTS","BRT",
    "PLD","DRE","FR","EGP","STAG","TRNO","REXR","IIPR","COLD","GOOD","NLCP",
    "O","NNN","SRC","EPRT","ADC","PINE","GTY","NTST","VICI","MPW","GMRE","CHCT",
    "BXP","SLG","KRC","HIW","PKY","CUZ","PDM","DEA","JBGS","VNO","SFO","CLI",
    "SPG","MAC","TCO","BRX","KIM","REG","AKR","CBL","PREIT","WPG","WPT",
    "WELL","VTR","OHI","LTC","HR","SNH","CTRE","NHI","SBRA","CSQ","SHO","UHT",
    "HST","PK","RHP","SHO","CHSP","APLE","CLDT","CHATM","INN","XENIA","BRAEMAR",
    "SVC","ILPT","GMRE","LAND","AFIN","EPRT","BNL","NTST","RTL","NLCP",
    # Utilities
    "NEE","SO","DUK","AEP","EXC","D","ED","FE","PCG","EIX","PEG","PPL","CMS","NI",
    "AES","ETR","WEC","DTE","LNT","EVRG","AEE","PNW","XEL","IDA","AVA","NWE","POR",
    "SRE","ES","CNP","OGE","MGEE","AWR","CWT","MSEX","SJW","YORW","ARTNA","GWRS",
    "AWK","WTR","WTRG","CWCO","GWRS","ARTNA","MSEX","YORW","SJW","CWT","AWR",
    # Communication Services (NYSE-listed)
    "T","VZ","LUMN","CTL","CNSL","SHEN","LMSA","LBRDKA","CABO","ATUS","WOW","LBRDK",
    "DIS","FOX","FOXA","NWSA","NWS","CBS","VIAC","OMC","IPG","PUB","GCI","MDP","NYT",
    "TWX","DISCA","DISCK","DISCB","AMC","CNK","RGC","IMAX","LGF-A","LGF-B","MCS",
]


# ── Market Gate ───────────────────────────────────────────────────────────────

def check_market_gate() -> tuple[bool, str]:
    proxies = ["SPY", "QQQ"]
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
    return True, "SPY and QQQ both above 50-day SMA"


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
    tickers = get_nyse_tickers()
    print(f"      {len(tickers)} NYSE stocks")

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
