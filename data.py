"""
data.py - Data fetching for Meridian Markets
=============================================
Architecture:
  - Alpaca API    → prices, bars, heatmap, indices (real-time, fast)
  - yfinance      → fundamentals, earnings, news (cached 4hrs — changes slowly)
  - WebSocket     → live streaming prices pushed to connected browsers
  - In-memory cache with TTLs at every layer

Cache TTLs:
  - Prices (Alpaca bars):   60 seconds
  - Live prices (WS):       real-time
  - Fundamentals (yf):      4 hours
  - Earnings (yf):          4 hours
  - News (yf):              15 minutes
  - Sector heatmap:         60 seconds
  - Macro summary:          60 seconds
  - Indices chart:          30 seconds
"""

import os
import time
import threading
import datetime as dt
from datetime import datetime, timedelta

# ============================================================
# CACHE STORE
# ============================================================
_cache      = {}   # key → (data, expires_at)
_cache_lock = threading.Lock()

def _get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry and time.time() < entry[1]:
            return entry[0]
        return None

def _set(key, data, ttl_seconds):
    with _cache_lock:
        _cache[key] = (data, time.time() + ttl_seconds)

# ============================================================
# LIVE PRICE STORE (updated by WebSocket)
# ============================================================
live_prices   = {}   # ticker → {"price": float, "change_pct": float, "updated": timestamp}
_ws_thread    = None
_ws_running   = False
_ws_lock      = threading.Lock()

# ============================================================
# STATIC DATA
# ============================================================
LIMITED_FUNDAMENTALS = ["IBIT", "IONQ", "NBIS", "IREN"]

SECTOR_MAP = {
    "XOM": "Energy", "SHEL": "Energy", "MPC": "Energy", "OKE": "Energy",
    "MSFT": "Technology", "AAPL": "Technology", "NVDA": "Technology",
    "AMD": "Technology", "SAP": "Technology", "PLTR": "Technology",
    "ILMN": "Technology", "IONQ": "Technology", "IREN": "Technology", "NBIS": "Technology",
    "AMZN": "Technology", "GOOG": "Technology", "GOOGL": "Technology", "TSLA": "Technology",
    "JNJ": "Healthcare", "MDT": "Healthcare", "EW": "Healthcare",
    "REGN": "Healthcare", "LLY": "Healthcare",
    "MA": "Financial", "AXP": "Financial", "PGR": "Financial", "HBAN": "Financial",
    "NUE": "Materials", "HII": "Defense", "DAL": "Airlines",
    "PENN": "Gaming", "MNST": "Consumer", "IBIT": "Crypto",
    "META": "Technology", "NFLX": "Technology", "BRKB": "Financial",
    "JPM": "Financial", "V": "Financial", "WMT": "Consumer", "DIS": "Consumer",
}

HEATMAP_EXTRAS = ["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","NFLX","JPM","V"]

SECTOR_STOCKS = {
    "Technology":       ["AAPL","MSFT","NVDA","GOOGL","META","AMZN","TSLA","AMD","PLTR","CRM"],
    "Financials":       ["JPM","BAC","WFC","GS","MS","V","MA","AXP","BLK","SCHW"],
    "Healthcare":       ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","MDT","REGN","PFE"],
    "Energy":           ["XOM","CVX","COP","SLB","MPC","PSX","OXY","VLO","HAL","EOG"],
    "Consumer Staples": ["WMT","PG","KO","PEP","COST","PM","MO","CL","GIS","KHC"],
    "Consumer Discr.":  ["AMZN","TSLA","HD","MCD","NKE","LOW","SBUX","TJX","BKNG","CMG"],
    "Industrials":      ["GE","CAT","HON","UPS","BA","LMT","RTX","DE","MMM","FDX"],
    "Materials":        ["LIN","APD","SHW","FCX","NEM","NUE","DOW","ECL","ALB","VMC"],
    "Communication":    ["GOOGL","META","NFLX","DIS","CMCSA","T","VZ","CHTR","PARA","WBD"],
    "Utilities":        ["NEE","DUK","SO","D","AEP","EXC","XEL","SRE","ES","ETR"],
    "Real Estate":      ["AMT","PLD","CCI","EQIX","PSA","O","WELL","DLR","AVB","EQR"],
    "Defense":          ["LMT","RTX","NOC","GD","BA","HII","TDG","LDOS","CACI","L3T"],
    "Airlines":         ["DAL","UAL","AAL","LUV","ALK","JBLU","HA","SAVE","ULCC","MESA"],
    "Biotech":          ["ABBV","REGN","VRTX","MRNA","BIIB","ILMN","ALNY","SGEN","BMRN","RARE"],
    "Gold":             ["GLD","GDX","NEM","GOLD","AEM","WPM","AGI","KGC","HMY","AU"],
    "Silver":           ["SLV","SILJ","WPM","PAAS","MAG","CDE","HL","SSRM","FSM","MTA"],
    "Oil":              ["USO","XOM","CVX","COP","OXY","SLB","HAL","MPC","VLO","PSX"],
    "Crypto (BTC)":     ["IBIT","COIN","MSTR","MARA","RIOT","CLSK","BTBT","CORZ","HUT","WULF"],
}

SECTOR_ETFS = {
    "XLK":"Technology","XLV":"Healthcare","XLE":"Energy","XLF":"Financials",
    "XLP":"Consumer Staples","XLY":"Consumer Discretionary","XLB":"Materials",
    "XLI":"Industrials","XLRE":"Real Estate","XLU":"Utilities","XLC":"Communication",
    "ITA":"Defense","JETS":"Airlines","IBB":"Biotech",
    "GLD":"Gold","SLV":"Silver","USO":"Oil","IBIT":"Crypto (BTC)",
}

MACRO_TICKERS = {
    "SPY":"S&P 500","QQQ":"NASDAQ 100","DIA":"Dow Jones",
    "TLT":"20yr Treasury","GLD":"Gold","UUP":"US Dollar","^VIX":"VIX"
}

INDEX_TICKERS = {"SPY":"S&P 500","QQQ":"NASDAQ 100","DIA":"Dow Jones"}


# ============================================================
# ALPACA CLIENT HELPERS
# ============================================================
def _alpaca_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    )

def _alpaca_bars(ticker, days=365, timeframe_str="day"):
    """Fetch daily bars from Alpaca for a single ticker. Returns DataFrame."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    import pandas as pd

    tf  = TimeFrame.Day if timeframe_str == "day" else TimeFrame.Minute
    end   = datetime.now()
    start = end - timedelta(days=days)

    client = _alpaca_data_client()
    bars   = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=tf,
        start=start,
        end=end,
    ))
    df = bars.df
    if df is None or df.empty:
        return None
    if hasattr(df.index, 'levels'):
        df = df.xs(ticker, level='symbol')
    return df

def _alpaca_multi_bars(tickers, days=2):
    """Fetch bars for multiple tickers in one Alpaca API call — much faster than looping."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    end   = datetime.now()
    start = end - timedelta(days=days + 5)  # buffer for weekends/holidays

    client = _alpaca_data_client()
    bars   = client.get_stock_bars(StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
    ))
    df = bars.df
    if df is None or df.empty:
        return {}

    results = {}
    for ticker in tickers:
        try:
            if hasattr(df.index, 'levels'):
                sub = df.xs(ticker, level='symbol')
            else:
                sub = df[df.index == ticker]
            if not sub.empty:
                results[ticker] = sub
        except:
            pass
    return results


# ============================================================
# PRICE DATA (Alpaca → real-time, 60s cache)
# ============================================================
def get_price_data(ticker):
    """
    Get current price + technical indicators for a ticker.
    Uses Alpaca for real-time data. Falls back to yfinance if Alpaca not configured.
    """
    cache_key = f"price:{ticker}"
    cached = _get(cache_key)
    if cached:
        return cached

    try:
        import numpy as np
        df = _alpaca_bars(ticker, days=400)
        if df is None or len(df) < 2:
            raise ValueError("Insufficient data")

        closes  = df["close"].values.tolist()
        volumes = df["volume"].values.tolist()
        curr    = closes[-1]
        prev    = closes[-2]
        change  = curr - prev
        change_pct = (change / prev) * 100

        ma50  = round(float(np.mean(closes[-50:])),  2) if len(closes) >= 50  else None
        ma200 = round(float(np.mean(closes[-200:])), 2) if len(closes) >= 200 else None

        high_52w = round(float(max(df["high"].values[-252:])),  2)
        low_52w  = round(float(min(df["low"].values[-252:])),   2)

        volume     = volumes[-1]
        avg_volume = float(np.mean(volumes[-20:]))
        vol_ratio  = round(volume / avg_volume, 2) if avg_volume > 0 else None

        if vol_ratio:
            if vol_ratio >= 2.0:   vol_signal = "Very High"
            elif vol_ratio >= 1.5: vol_signal = "High"
            elif vol_ratio >= 0.8: vol_signal = "Normal"
            else:                  vol_signal = "Low"
        else:
            vol_signal = "N/A"

        if ma50 and ma200:
            if curr > ma50 and curr > ma200:   ma_context = "Above both MAs - bullish"
            elif curr < ma50 and curr < ma200: ma_context = "Below both MAs - bearish"
            elif curr > ma50 and curr < ma200: ma_context = "Above 50MA, below 200MA - recovering"
            else:                              ma_context = "Below 50MA, above 200MA - weakening"
        else:
            ma_context = "N/A"

        result = {
            "ticker":     ticker,
            "price":      round(curr, 2),
            "change":     round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume":     f"{int(volume):,}",
            "volume_raw": int(volume),
            "avg_volume": f"{int(avg_volume):,}",
            "vol_ratio":  vol_ratio,
            "vol_signal": vol_signal,
            "high_52w":   high_52w,
            "low_52w":    low_52w,
            "ma50":       ma50,
            "ma200":      ma200,
            "ma_context": ma_context,
            "market_cap": 0,   # filled by fundamentals cache
            "source":     "alpaca",
        }
        _set(cache_key, result, ttl_seconds=60)
        return result

    except Exception as alpaca_err:
        # Fallback to yfinance if Alpaca not configured or fails
        try:
            import yfinance as yf
            import numpy as np
            stock   = yf.Ticker(ticker)
            hist    = stock.history(period="1y")
            if len(hist) < 2:
                return {"ticker": ticker, "error": "No data"}
            closes  = hist["Close"].values.tolist()
            volumes = hist["Volume"].values.tolist()
            curr    = closes[-1]
            prev    = closes[-2]
            ma50    = round(float(np.mean(closes[-50:])), 2) if len(closes) >= 50 else None
            ma200   = round(float(np.mean(closes[-200:])), 2) if len(closes) >= 200 else None
            vol_ratio = round(volumes[-1] / float(np.mean(volumes)), 2)
            vol_signal = "High" if vol_ratio >= 1.5 else "Normal"
            ma_context = "N/A"
            if ma50 and ma200:
                if curr > ma50 and curr > ma200: ma_context = "Above both MAs - bullish"
                elif curr < ma50 and curr < ma200: ma_context = "Below both MAs - bearish"
            result = {
                "ticker": ticker, "price": round(curr, 2),
                "change": round(curr - prev, 2),
                "change_pct": round(((curr-prev)/prev)*100, 2),
                "volume": f"{int(volumes[-1]):,}", "volume_raw": int(volumes[-1]),
                "avg_volume": f"{int(np.mean(volumes)):,}",
                "vol_ratio": vol_ratio, "vol_signal": vol_signal,
                "high_52w": round(max(hist["High"].values), 2),
                "low_52w":  round(min(hist["Low"].values),  2),
                "ma50": ma50, "ma200": ma200, "ma_context": ma_context,
                "market_cap": 0, "source": "yfinance_fallback",
            }
            _set(cache_key, result, ttl_seconds=60)
            return result
        except Exception as e:
            return {"ticker": ticker, "error": str(e)}


# ============================================================
# FUNDAMENTALS (yfinance, 4hr cache — changes slowly)
# ============================================================
def get_fundamentals(ticker):
    if ticker in LIMITED_FUNDAMENTALS:
        return {"ticker": ticker, "note": "Limited fundamental data"}

    cache_key = f"fund:{ticker}"
    cached = _get(cache_key)
    if cached:
        return cached

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info  = stock.info

        eps            = info.get("trailingEps",          "N/A")
        pe_ratio       = info.get("trailingPE",           "N/A")
        forward_pe     = info.get("forwardPE",            "N/A")
        profit_margin  = info.get("profitMargins",        "N/A")
        revenue_growth = info.get("revenueGrowth",        "N/A")
        roe            = info.get("returnOnEquity",       "N/A")
        market_cap     = info.get("marketCap",            "N/A")
        debt_to_equity = info.get("debtToEquity",         "N/A")
        sector         = info.get("sector",               SECTOR_MAP.get(ticker, "Unknown"))
        company_name   = info.get("longName",             ticker)
        short_ratio    = info.get("shortRatio",           "N/A")
        short_pct      = info.get("shortPercentOfFloat",  "N/A")
        inst_ownership = info.get("institutionalOwnershipPercentage", "N/A")
        beta           = info.get("beta",                 "N/A")

        if isinstance(profit_margin,  float): profit_margin  = f"{profit_margin  * 100:.1f}%"
        if isinstance(revenue_growth, float): revenue_growth = f"{revenue_growth * 100:.1f}%"
        if isinstance(roe,            float): roe            = f"{roe            * 100:.1f}%"
        if isinstance(short_pct,      float): short_pct      = f"{short_pct      * 100:.1f}%"
        if isinstance(inst_ownership, float): inst_ownership = f"{inst_ownership * 100:.1f}%"
        if isinstance(market_cap, (int, float)): market_cap  = f"${market_cap   / 1e9:.1f}B"

        if isinstance(short_pct, str) and "%" in short_pct:
            sp = float(short_pct.replace("%",""))
            if sp > 20:   short_signal = "Very High Short Interest"
            elif sp > 10: short_signal = "High Short Interest"
            elif sp > 5:  short_signal = "Moderate Short Interest"
            else:         short_signal = "Low Short Interest"
        else:
            short_signal = "N/A"

        try:
            options_dates = stock.options
            has_options   = len(options_dates) > 0 if options_dates else False
            next_expiry   = options_dates[0] if has_options else None
        except:
            has_options = False
            next_expiry = None

        score = 0
        if isinstance(pe_ratio, float) and pe_ratio < 25: score += 1
        if isinstance(profit_margin, str) and "%" in profit_margin:
            if float(profit_margin.replace("%","")) > 10: score += 1
        if isinstance(revenue_growth, str) and "%" in revenue_growth:
            if float(revenue_growth.replace("%","")) > 5: score += 1
        if isinstance(eps, float) and eps > 0: score += 1
        if isinstance(roe, str) and "%" in roe:
            if float(roe.replace("%","")) > 15: score += 1

        signal = "FAVORABLE" if score >= 4 else "NEUTRAL" if score >= 2 else "CAUTION"

        result = {
            "ticker": ticker, "company_name": company_name, "sector": sector,
            "eps": eps, "pe_ratio": pe_ratio, "forward_pe": forward_pe,
            "profit_margin": profit_margin, "revenue_growth": revenue_growth,
            "roe": roe, "market_cap": market_cap, "debt_to_equity": debt_to_equity,
            "signal": signal,
            "short_pct": short_pct, "short_ratio": short_ratio, "short_signal": short_signal,
            "inst_ownership": inst_ownership, "beta": beta,
            "has_options": has_options, "next_expiry": next_expiry,
            "source": "yfinance_cached",
        }
        _set(cache_key, result, ttl_seconds=4 * 3600)   # 4 hours
        return result
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


# ============================================================
# EARNINGS (yfinance, 4hr cache)
# ============================================================
def get_earnings_date(ticker):
    cache_key = f"earnings:{ticker}"
    cached = _get(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        info      = yf.Ticker(ticker).info
        timestamp = info.get("earningsTimestamp", None)
        if timestamp:
            earnings_date = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).date()
            today         = dt.date.today()
            days_away     = (earnings_date - today).days
            if 0 <= days_away <= 14:
                label  = "TODAY" if days_away == 0 else "Tomorrow" if days_away == 1 else f"In {days_away} days"
                hour   = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).hour
                timing = "Before Open" if hour < 12 else "After Close" if hour >= 16 else "During Market"
                result = {
                    "date":      earnings_date.strftime("%A, %B %d"),
                    "label":     label,
                    "days_away": days_away,
                    "timing":    timing
                }
                _set(cache_key, result, ttl_seconds=4 * 3600)
                return result
        _set(cache_key, None, ttl_seconds=4 * 3600)
        return None
    except:
        return None


# ============================================================
# NEWS (yfinance, 15min cache)
# ============================================================
def get_news(ticker, max_headlines=5):
    cache_key = f"news:{ticker}"
    cached = _get(cache_key)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
        news      = yf.Ticker(ticker).news
        headlines = []
        for item in news[:max_headlines]:
            if "content" in item:
                title = item["content"].get("title", "")
                link  = item["content"].get("canonicalUrl", {}).get("url", "#")
            else:
                title = item.get("title", "")
                link  = item.get("link",  "#")
            if not title:
                continue
            title = ''.join(c if ord(c) < 128 else ' ' for c in title)
            headlines.append({"title": title, "link": link})
        _set(cache_key, headlines, ttl_seconds=15 * 60)
        return headlines
    except:
        return []


# ============================================================
# HEATMAP (Alpaca multi-fetch, 60s cache)
# ============================================================
def get_heatmap_data(watchlist):
    """Get price data for watchlist + market extras. Single Alpaca call for all tickers."""
    cache_key = f"heatmap:{','.join(sorted(watchlist))}"
    cached = _get(cache_key)
    if cached:
        return cached

    all_tickers = list(dict.fromkeys(watchlist + HEATMAP_EXTRAS))

    try:
        bars_by_ticker = _alpaca_multi_bars(all_tickers, days=5)
        results = []

        for ticker in all_tickers:
            try:
                df = bars_by_ticker.get(ticker)
                if df is None or len(df) < 2:
                    continue
                curr       = float(df["close"].iloc[-1])
                prev       = float(df["close"].iloc[-2])
                change_pct = round(((curr - prev) / prev) * 100, 2)

                fund = get_fundamentals(ticker)   # served from 4hr cache — free
                sector     = fund.get("sector", SECTOR_MAP.get(ticker, "Other")) if fund else SECTOR_MAP.get(ticker, "Other")
                name       = fund.get("company_name", ticker) if fund else ticker
                market_cap_raw = fund.get("market_cap", "0") if fund else "0"
                # Convert "$2.5T" / "$500B" style back to numeric for sizing
                market_cap = _parse_market_cap(market_cap_raw)

                results.append({
                    "ticker":       ticker,
                    "name":         name[:30],
                    "price":        round(curr, 2),
                    "change_pct":   change_pct,
                    "market_cap":   market_cap,
                    "sector":       sector,
                    "in_watchlist": ticker in watchlist,
                })
            except:
                pass

        _set(cache_key, results, ttl_seconds=60)
        return results

    except Exception as e:
        # Fallback to yfinance if Alpaca down
        import yfinance as yf
        results = []
        for ticker in all_tickers:
            try:
                hist = yf.Ticker(ticker).history(period="2d")
                if len(hist) < 2:
                    continue
                curr       = float(hist["Close"].iloc[-1])
                prev       = float(hist["Close"].iloc[-2])
                change_pct = round(((curr - prev) / prev) * 100, 2)
                results.append({
                    "ticker": ticker, "name": ticker,
                    "price": round(curr, 2), "change_pct": change_pct,
                    "market_cap": 0, "sector": SECTOR_MAP.get(ticker, "Other"),
                    "in_watchlist": ticker in watchlist,
                })
            except:
                pass
        _set(cache_key, results, ttl_seconds=60)
        return results


def _parse_market_cap(raw):
    """Convert '$2.5T' / '$500.0B' string back to integer for heatmap sizing."""
    if isinstance(raw, (int, float)):
        return int(raw)
    if not isinstance(raw, str):
        return 0
    raw = raw.replace("$","").strip()
    try:
        if raw.endswith("T"):   return int(float(raw[:-1]) * 1e12)
        elif raw.endswith("B"): return int(float(raw[:-1]) * 1e9)
        elif raw.endswith("M"): return int(float(raw[:-1]) * 1e6)
        return int(float(raw))
    except:
        return 0


# ============================================================
# SECTOR HEATMAP (Alpaca multi-fetch, 60s cache)
# ============================================================
def get_sector_heatmap():
    cache_key = "sector_heatmap"
    cached = _get(cache_key)
    if cached:
        return cached

    etf_tickers = list(SECTOR_ETFS.keys())
    try:
        bars_by_ticker = _alpaca_multi_bars(etf_tickers, days=5)
        results = []
        for ticker, label in SECTOR_ETFS.items():
            df = bars_by_ticker.get(ticker)
            if df is None or len(df) < 2:
                continue
            curr       = float(df["close"].iloc[-1])
            prev       = float(df["close"].iloc[-2])
            change_pct = round(((curr - prev) / prev) * 100, 2)
            results.append({
                "ticker": ticker, "label": label,
                "price": round(curr, 2), "change_pct": change_pct,
            })
        _set(cache_key, results, ttl_seconds=60)
        return results
    except:
        # Fallback
        import yfinance as yf
        results = []
        for ticker, label in SECTOR_ETFS.items():
            try:
                hist = yf.Ticker(ticker).history(period="2d")
                if len(hist) < 2: continue
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                results.append({"ticker": ticker, "label": label,
                    "price": round(curr, 2),
                    "change_pct": round(((curr-prev)/prev)*100, 2)})
            except: pass
        _set(cache_key, results, ttl_seconds=60)
        return results


# ============================================================
# MACRO SUMMARY (60s cache)
# ============================================================
def get_macro_summary():
    cache_key = "macro"
    cached = _get(cache_key)
    if cached:
        return cached

    tickers = list(MACRO_TICKERS.keys())
    try:
        bars_by_ticker = _alpaca_multi_bars(
            [t for t in tickers if not t.startswith("^")], days=5
        )
        results = []
        for ticker, label in MACRO_TICKERS.items():
            try:
                if ticker.startswith("^"):
                    # VIX — yfinance only (Alpaca doesn't carry ^VIX)
                    import yfinance as yf
                    hist = yf.Ticker(ticker).history(period="2d")
                    if len(hist) >= 1:
                        results.append({"label": label, "ticker": ticker,
                            "price": round(float(hist["Close"].iloc[-1]), 2),
                            "change_pct": 0})
                    continue
                df = bars_by_ticker.get(ticker)
                if df is None or len(df) < 2:
                    continue
                curr = float(df["close"].iloc[-1])
                prev = float(df["close"].iloc[-2])
                results.append({"label": label, "ticker": ticker,
                    "price": round(curr, 2),
                    "change_pct": round(((curr-prev)/prev)*100, 2)})
            except:
                pass
        _set(cache_key, results, ttl_seconds=60)
        return results
    except Exception as e:
        return []


# ============================================================
# INDICES CHART (Alpaca intraday bars, 30s cache)
# ============================================================
def get_indices_chart_data():
    cache_key = "indices_chart"
    cached = _get(cache_key)
    if cached:
        return cached

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    now          = datetime.now()
    is_weekday   = now.weekday() < 5
    market_open  = is_weekday and (now.hour > 9 or (now.hour == 9 and now.minute >= 30)) and now.hour < 16

    tickers = list(INDEX_TICKERS.keys())
    results = {}

    try:
        client = _alpaca_data_client()
        # Use 5-minute bars for intraday
        end   = datetime.now()
        start = end - timedelta(days=2)

        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Minute if market_open else TimeFrame.Day,
            start=start,
            end=end,
        ))
        df_all = bars.df

        for ticker, label in INDEX_TICKERS.items():
            try:
                if hasattr(df_all.index, 'levels'):
                    df = df_all.xs(ticker, level='symbol')
                else:
                    df = df_all

                if not market_open:
                    # Just last trading day
                    last_date = df.index[-1].date()
                    df = df[df.index.date == last_date]

                prices     = [round(float(p), 2) for p in df["close"].tolist()]
                timestamps = [str(t) for t in df.index.tolist()]
                open_price = float(df["close"].iloc[0])
                curr_price = float(df["close"].iloc[-1])
                change_pct = round(((curr_price - open_price) / open_price) * 100, 2) if open_price > 0 else 0

                results[ticker] = {
                    "label": label, "prices": prices, "timestamps": timestamps,
                    "current": curr_price, "open": open_price, "change_pct": change_pct,
                }
            except Exception as e:
                results[ticker] = {"label": label, "error": str(e)}

        _set(cache_key, results, ttl_seconds=30)
        return results

    except Exception:
        # Fallback to yfinance
        import yfinance as yf
        for ticker, label in INDEX_TICKERS.items():
            try:
                stock = yf.Ticker(ticker)
                hist  = stock.history(period="1d", interval="5m")
                if len(hist) == 0:
                    hist = stock.history(period="2d", interval="5m")
                prices     = [round(float(p), 2) for p in hist["Close"].tolist()]
                timestamps = [str(t) for t in hist.index.tolist()]
                open_price = float(hist["Close"].iloc[0]) if len(hist) > 0 else 0
                curr_price = float(hist["Close"].iloc[-1]) if len(hist) > 0 else 0
                change_pct = round(((curr_price - open_price) / open_price) * 100, 2) if open_price > 0 else 0
                results[ticker] = {"label": label, "prices": prices,
                    "timestamps": timestamps, "current": curr_price,
                    "open": open_price, "change_pct": change_pct}
            except Exception as e:
                results[ticker] = {"label": label, "error": str(e)}
        _set(cache_key, results, ttl_seconds=30)
        return results


# ============================================================
# SECTOR STOCKS (Alpaca, 60s cache)
# ============================================================
def get_sector_stocks(sector_name):
    cache_key = f"sector_stocks:{sector_name}"
    cached = _get(cache_key)
    if cached:
        return cached

    tickers = SECTOR_STOCKS.get(sector_name, SECTOR_STOCKS.get("Technology"))[:10]
    try:
        bars_by_ticker = _alpaca_multi_bars(tickers, days=5)
        results = []
        for ticker in tickers:
            try:
                df = bars_by_ticker.get(ticker)
                if df is None or len(df) < 2:
                    continue
                curr       = float(df["close"].iloc[-1])
                prev       = float(df["close"].iloc[-2])
                change_pct = round(((curr - prev) / prev) * 100, 2)
                # Recent prices for sparkline
                prices     = [round(float(p), 2) for p in df["close"].tail(20).tolist()]

                fund = get_fundamentals(ticker)   # free from 4hr cache
                name       = fund.get("company_name", ticker) if fund else ticker
                market_cap = _parse_market_cap(fund.get("market_cap", "0") if fund else "0")

                results.append({
                    "ticker": ticker, "name": name[:30],
                    "price": round(curr, 2), "change_pct": change_pct,
                    "market_cap": market_cap, "prices": prices,
                })
            except:
                pass
        _set(cache_key, results, ttl_seconds=60)
        return results
    except:
        import yfinance as yf
        results = []
        for ticker in tickers:
            try:
                hist = yf.Ticker(ticker).history(period="2d")
                if len(hist) < 2: continue
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                results.append({"ticker": ticker, "name": ticker,
                    "price": round(curr, 2),
                    "change_pct": round(((curr-prev)/prev)*100, 2),
                    "market_cap": 0,
                    "prices": [round(float(p),2) for p in hist["Close"].tail(20)]})
            except: pass
        _set(cache_key, results, ttl_seconds=60)
        return results


# ============================================================
# MARKET NEWS (RSS, 15min cache)
# ============================================================
def get_market_news():
    cache_key = "market_news"
    cached = _get(cache_key)
    if cached:
        return cached

    import urllib.request
    import xml.etree.ElementTree as ET
    from concurrent.futures import ThreadPoolExecutor, as_completed

    feeds = {
        "CNBC":         "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
        "MarketWatch":  "https://feeds.marketwatch.com/marketwatch/topstories",
        "Yahoo Finance":"https://finance.yahoo.com/news/rssindex",
        "Investopedia": "https://www.investopedia.com/feedbuilder/feed/getfeed?feedName=rss_headline",
        "Seeking Alpha":"https://seekingalpha.com/market_currents.xml",
    }

    def fetch_feed(source, url):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Accept":     "application/rss+xml, application/xml, text/xml, */*"
            })
            with urllib.request.urlopen(req, timeout=5) as resp:
                root = ET.fromstring(resp.read())
            items = []
            for item in root.findall(".//item")[:5]:
                title = item.findtext("title", "").strip()
                link  = item.findtext("link",  "").strip() or item.findtext("guid", "#").strip()
                pub   = item.findtext("pubDate", "").strip()
                if title and link and len(title) > 10:
                    title = ''.join(c if ord(c) < 128 else ' ' for c in title)
                    if title.lower() not in ["top stories", "markets", "business", "finance"]:
                        items.append({"source": source, "title": title,
                                      "link": link, "pubDate": pub})
            return items
        except:
            return []

    all_news = []
    # Fetch all feeds in parallel — one slow/dead feed won't block others
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_feed, src, url): src for src, url in feeds.items()}
        for future in as_completed(futures, timeout=8):
            try:
                all_news.extend(future.result())
            except:
                pass

    _set(cache_key, all_news, ttl_seconds=15 * 60)
    return all_news


# ============================================================
# INDUSTRY NEWS (15min cache)
# ============================================================
def get_industry_news(sector):
    sector_etfs = {
        "Technology":"XLK","Healthcare":"XLV","Energy":"XLE",
        "Financial":"XLF","Consumer":"XLP","Materials":"XLB",
        "Defense":"ITA","Airlines":"JETS","Gaming":"BETZ","Crypto":"IBIT"
    }
    etf = sector_etfs.get(sector, "SPY")
    return get_news(etf, max_headlines=5)


# ============================================================
# WEBSOCKET STREAMING (live prices pushed to browser)
# ============================================================
def start_price_stream(tickers):
    """
    Start Alpaca WebSocket stream for given tickers.
    Updates live_prices dict in real-time.
    Call this once at app startup with your watchlist universe.
    """
    global _ws_thread, _ws_running

    if _ws_running:
        return   # already running

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("Alpaca keys not set — WebSocket stream not started")
        return

    def _stream_worker():
        global _ws_running
        _ws_running = True
        try:
            from alpaca.data.live import StockDataStream
            stream = StockDataStream(api_key, secret_key)

            async def on_bar(bar):
                """Fires every minute with the latest bar for each subscribed ticker."""
                with _ws_lock:
                    prev = live_prices.get(bar.symbol, {})
                    prev_price = prev.get("price", bar.close)
                    change_pct = round(((bar.close - prev_price) / prev_price) * 100, 2) if prev_price else 0
                    live_prices[bar.symbol] = {
                        "price":      round(float(bar.close), 2),
                        "change_pct": change_pct,
                        "volume":     int(bar.volume),
                        "updated":    time.time(),
                    }

            stream.subscribe_bars(on_bar, *tickers)
            print(f"Alpaca WebSocket stream started — tracking {len(tickers)} tickers")
            stream.run()
        except Exception as e:
            print(f"WebSocket stream error: {e}")
            _ws_running = False

    import asyncio
    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_stream_worker_async())

    async def _stream_worker_async():
        global _ws_running
        _ws_running = True
        try:
            from alpaca.data.live import StockDataStream
            api_key    = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")
            stream     = StockDataStream(api_key, secret_key)

            async def on_bar(bar):
                with _ws_lock:
                    live_prices[bar.symbol] = {
                        "price":      round(float(bar.close), 2),
                        "change_pct": 0,
                        "volume":     int(bar.volume),
                        "updated":    time.time(),
                    }

            stream.subscribe_bars(on_bar, *tickers)
            print(f"✅ Alpaca WebSocket started — {len(tickers)} tickers")
            await stream._run_forever()
        except Exception as e:
            print(f"WebSocket error: {e}")
            _ws_running = False

    _ws_thread = threading.Thread(target=_run, daemon=True)
    _ws_thread.start()


def get_live_price(ticker):
    """Get the most recently streamed price for a ticker, or fall back to REST."""
    with _ws_lock:
        entry = live_prices.get(ticker)
    if entry and time.time() - entry.get("updated", 0) < 120:
        return entry
    # Fall back to REST if WebSocket hasn't updated recently
    return get_price_data(ticker)


def get_live_prices_batch(tickers):
    """Return live prices for multiple tickers at once — used by /api/live-prices SSE endpoint."""
    result = {}
    with _ws_lock:
        for ticker in tickers:
            entry = live_prices.get(ticker)
            if entry and time.time() - entry.get("updated", 0) < 120:
                result[ticker] = entry
    return result