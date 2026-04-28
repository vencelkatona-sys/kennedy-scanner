"""
Kennedy Channel Scanner
-----------------------
Automatikusan keres Elliott-hullám struktúrát + Kennedy-csatorna belépési pontokat.
Feltételek: MACD bullish keresztezés + RSI oversold + csatorna alsó zónájában
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import schedule
from datetime import datetime
import os

# ============================================================
# BEÁLLÍTÁSOK — itt módosíthatod a paramétereket
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "IDE_IRD_A_TOKEN-T")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "IDE_IRD_A_CHAT_ID-T")

# Timeframe: "1wk" = heti, "1d" = napi
TIMEFRAMES = ["1wk", "1d"]

# Hány gyertya alapján keressen pivot pontokat
PIVOT_LOOKBACK = 10  # heti charthoz jobb a nagyobb szám

# RSI oversold határ
RSI_OVERSOLD = 42

# Részvénylista — kezdjük S&P 500 egy részével + néhány európai
SP500_SAMPLE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    "AMD", "INTC", "CRM", "ADBE", "PYPL", "SHOP", "SQ", "COIN",
    "ALGN", "ISRG", "DXCM", "IDXX", "PODD", "MTCH", "SNAP", "PINS",
    "UBER", "LYFT", "ABNB", "DASH", "RBLX", "U", "PLTR", "SNOW",
    "DDOG", "NET", "ZS", "CRWD", "S", "FRSH", "PATH", "SYM",
    "OSCR", "ROOT", "OPFI", "PGY", "RXRX", "SDGR", "ALMU",
    "NICE", "FIVN", "CXM", "EVTC",
    # Európai ADR-ek
    "ASML", "SAP", "NVO", "AZN", "SHEL", "BP", "RIO", "BHP",
]

# ============================================================
# SEGÉDFÜGGVÉNYEK
# ============================================================

def send_telegram(message: str):
    """Telegram üzenet küldése."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"Telegram hiba: {e}")
        return False


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI számítás."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD számítás. Visszaad: macd_line, signal_line, histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def find_pivots(df: pd.DataFrame, lookback: int = 10):
    """
    Lokális csúcsok és völgyek keresése.
    Visszaad: pivot_highs, pivot_lows (index listák)
    """
    highs = []
    lows = []
    closes = df["Close"].values

    for i in range(lookback, len(closes) - lookback):
        window = closes[i - lookback: i + lookback + 1]
        if closes[i] == max(window):
            highs.append(i)
        if closes[i] == min(window):
            lows.append(i)

    return highs, lows


def build_kennedy_channel(df: pd.DataFrame, highs: list, lows: list):
    """
    Kennedy-csatorna építése:
    - Felső vonal: (1) és (3) csúcson átmenő egyenes
    - Alsó vonal: párhuzamos, (2) völgyön átmenő
    - Középvonal: a kettő közepe
    
    Visszaad: (upper, middle, lower) az utolsó gyertyára vetítve, vagy None
    """
    closes = df["Close"].values
    n = len(closes)

    # Szükséges legalább 2 csúcs és 1 völgy
    if len(highs) < 2 or len(lows) < 1:
        return None

    # Legutóbbi 2 csúcs = (1) és (3)
    h1_idx = highs[-2]
    h3_idx = highs[-1]
    h1_val = closes[h1_idx]
    h3_val = closes[h3_idx]

    # Közbülső völgy (2) — h1 és h3 között
    lows_between = [l for l in lows if h1_idx < l < h3_idx]
    if not lows_between:
        return None
    h2_idx = lows_between[0]
    h2_val = closes[h2_idx]

    # Felső vonal meredeksége (h1 → h3)
    if h3_idx == h1_idx:
        return None
    slope = (h3_val - h1_val) / (h3_idx - h1_idx)

    # Vetítés az utolsó gyertyára
    last_idx = n - 1
    upper_last = h1_val + slope * (last_idx - h1_idx)
    lower_last = h2_val + slope * (last_idx - h2_idx)

    # Ha a csatorna "fordított" (lefelé megy), kihagyjuk
    if upper_last < lower_last:
        upper_last, lower_last = lower_last, upper_last

    middle_last = (upper_last + lower_last) / 2

    return {
        "upper": upper_last,
        "middle": middle_last,
        "lower": lower_last,
        "slope": slope,
        "h1_price": h1_val,
        "h3_price": h3_val,
        "h2_price": h2_val,
    }


def is_macd_bullish_cross(macd_line: pd.Series, signal_line: pd.Series, lookback: int = 3) -> bool:
    """
    Ellenőrzi, hogy az utolsó `lookback` gyertyán belül volt-e bullish MACD keresztezés.
    Bullish keresztezés: macd_line átmegy a signal_line fölé (alulról felfelé).
    """
    for i in range(1, lookback + 1):
        if (macd_line.iloc[-i] > signal_line.iloc[-i] and
                macd_line.iloc[-i - 1] <= signal_line.iloc[-i - 1]):
            return True
    return False


def analyze_symbol(symbol: str, timeframe: str) -> dict | None:
    """
    Egy részvény elemzése. Visszaad egy dict-et ha belépési feltételek teljesülnek,
    különben None.
    """
    try:
        period = "5y" if timeframe == "1wk" else "2y"
        df = yf.download(symbol, period=period, interval=timeframe,
                         progress=False, auto_adjust=True)

        if df is None or len(df) < 60:
            return None

        df = df.copy()
        closes = df["Close"].squeeze()

        # Indikátorok
        rsi = calculate_rsi(closes)
        macd_line, signal_line, histogram = calculate_macd(closes)

        current_rsi = rsi.iloc[-1]
        current_price = closes.iloc[-1]

        # RSI feltétel
        if current_rsi > RSI_OVERSOLD:
            return None

        # MACD bullish keresztezés (utóbbi 3 gyertyán belül)
        if not is_macd_bullish_cross(macd_line, signal_line, lookback=3):
            return None

        # Pivot keresés
        pivot_lookback = PIVOT_LOOKBACK if timeframe == "1wk" else 5
        highs, lows = find_pivots(df, lookback=pivot_lookback)

        # Kennedy-csatorna
        channel = build_kennedy_channel(df, highs, lows)
        if channel is None:
            return None

        lower = channel["lower"]
        middle = channel["middle"]
        upper = channel["upper"]

        # Ár a csatorna alsó zónájában? (alsó vonal + a csatorna 25%-án belül)
        channel_width = upper - lower
        if channel_width <= 0:
            return None

        zone_threshold = lower + channel_width * 0.25
        if current_price > zone_threshold:
            return None

        # Minden feltétel teljesül!
        return {
            "symbol": symbol,
            "timeframe": "Heti" if timeframe == "1wk" else "Napi",
            "price": round(float(current_price), 2),
            "rsi": round(float(current_rsi), 1),
            "macd": round(float(macd_line.iloc[-1]), 3),
            "signal": round(float(signal_line.iloc[-1]), 3),
            "channel_lower": round(float(lower), 2),
            "channel_middle": round(float(middle), 2),
            "channel_upper": round(float(upper), 2),
            "h1": round(float(channel["h1_price"]), 2),
            "h3": round(float(channel["h3_price"]), 2),
        }

    except Exception as e:
        print(f"  Hiba {symbol} ({timeframe}): {e}")
        return None


def format_telegram_message(result: dict) -> str:
    """Telegram üzenet formázása."""
    tf = result["timeframe"]
    sym = result["symbol"]
    tv_link = f"https://www.tradingview.com/chart/?symbol={sym}"

    msg = (
        f"🎯 <b>{sym}</b> — {tf} chart\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"💰 Ár: <b>${result['price']}</b>\n\n"
        f"✅ MACD bullish keresztezés\n"
        f"   MACD: {result['macd']} | Signal: {result['signal']}\n\n"
        f"✅ RSI: <b>{result['rsi']}</b> (oversold)\n\n"
        f"✅ Kennedy-csatorna alsó zónájában\n"
        f"   Alsó: ${result['channel_lower']}\n"
        f"   Közép (célár): ${result['channel_middle']}\n"
        f"   Felső: ${result['channel_upper']}\n\n"
        f"📊 Hullámcsúcsok: (1)=${result['h1']} → (3)=${result['h3']}\n\n"
        f"🔗 <a href='{tv_link}'>TradingView chart</a>\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    return msg


def run_scan():
    """Teljes scan futtatása."""
    print(f"\n{'='*50}")
    print(f"Scanner indítva: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")

    found = 0

    for symbol in SP500_SAMPLE:
        for timeframe in TIMEFRAMES:
            print(f"  Elemzés: {symbol} ({timeframe})...", end=" ")
            result = analyze_symbol(symbol, timeframe)

            if result:
                print(f"✅ TALÁLAT!")
                msg = format_telegram_message(result)
                send_telegram(msg)
                found += 1
                time.sleep(1)  # Telegram rate limit elkerülése
            else:
                print("—")

            time.sleep(0.3)  # Yahoo Finance rate limit

    summary = (
        f"📋 <b>Scan befejezve</b>\n"
        f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"🔍 Vizsgált részvények: {len(SP500_SAMPLE)}\n"
        f"✅ Találatok: {found}"
    )
    send_telegram(summary)
    print(f"\nScan kész. Találatok: {found}")


# ============================================================
# ÜTEMEZÉS — mikor fusson a scan?
# ============================================================

if __name__ == "__main__":
    print("Kennedy Channel Scanner indítva...")
    send_telegram("🚀 <b>Kennedy Scanner elindult!</b>\nHamarosan jön az első scan eredménye.")

    # Azonnali első futás
    run_scan()

    # Ütemezés: hétfő és csütörtök reggel 8:00 (heti chart)
    # + minden nap reggel 8:00 (napi chart)
    schedule.every().monday.at("08:00").do(run_scan)
    schedule.every().thursday.at("08:00").do(run_scan)

    print("Ütemezés aktív. Scanner fut a háttérben...")
    while True:
        schedule.run_pending()
        time.sleep(60)
