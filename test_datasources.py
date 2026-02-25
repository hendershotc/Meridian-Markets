"""
test_datasources.py
Run this locally to test both Yahoo Finance and Alpaca data.

Usage:
    python test_datasources.py

Make sure your .env file has ALPACA_API_KEY and ALPACA_SECRET_KEY set,
or export them as environment variables before running.
"""

import os
from dotenv import load_dotenv
load_dotenv()

TEST_TICKERS = ["AAPL", "MSFT", "NVDA"]

# ============================================================
# YAHOO FINANCE TEST
# ============================================================
def test_yfinance():
    print("\n" + "="*55)
    print("  YAHOO FINANCE TEST")
    print("="*55)
    try:
        import yfinance as yf
        import numpy as np

        for ticker in TEST_TICKERS:
            stock = yf.Ticker(ticker)
            hist  = stock.history(period="1y")

            if hist.empty:
                print(f"{ticker}: ❌ No data returned")
                continue

            closes  = hist["Close"].values.tolist()
            volumes = hist["Volume"].values.tolist()
            curr    = closes[-1]
            prev    = closes[-2]
            ma50    = np.mean(closes[-50:])
            ma200   = np.mean(closes[-200:]) if len(closes) >= 200 else None
            vol_avg = np.mean(volumes[-8:-1])
            vol_rat = volumes[-1] / vol_avg if vol_avg > 0 else 1

            print(f"\n{ticker}")
            print(f"  Price:        ${curr:.2f}  ({((curr-prev)/prev)*100:+.2f}% vs prev close)")
            print(f"  Data date:    {hist.index[-1].date()} (may be delayed 15-20min)")
            print(f"  Data points:  {len(closes)} trading days")
            print(f"  MA50:         ${ma50:.2f}  ({'ABOVE' if curr > ma50 else 'BELOW'})")
            print(f"  MA200:        ${ma200:.2f}  ({'ABOVE' if curr > ma200 else 'BELOW'})" if ma200 else "  MA200:        N/A")
            print(f"  Volume ratio: {vol_rat:.2f}x avg  ({'HIGH' if vol_rat > 1.1 else 'normal'})")

        print("\n✅ Yahoo Finance: working")
        print("⚠️  Note: prices are ~15-20 minutes delayed on free tier")

    except ImportError:
        print("❌ yfinance not installed — run: pip install yfinance")
    except Exception as e:
        print(f"❌ Yahoo Finance error: {e}")


# ============================================================
# ALPACA TEST
# ============================================================
def test_alpaca():
    print("\n" + "="*55)
    print("  ALPACA TEST")
    print("="*55)

    api_key    = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    is_paper   = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        print("❌ ALPACA_API_KEY / ALPACA_SECRET_KEY not set in .env")
        return

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta
        import numpy as np

        # Account check
        trading = TradingClient(api_key, secret_key, paper=is_paper)
        account = trading.get_account()
        print(f"\n✅ Connected to Alpaca {'PAPER' if is_paper else 'LIVE'} account")
        print(f"  Portfolio value: ${float(account.portfolio_value):,.2f}")
        print(f"  Cash:            ${float(account.cash):,.2f}")
        print(f"  Buying power:    ${float(account.buying_power):,.2f}")

        # Open positions
        positions = trading.get_all_positions()
        if positions:
            print(f"\n  Open positions ({len(positions)}):")
            for pos in positions:
                pnl = float(pos.unrealized_pl)
                pct = float(pos.unrealized_plpc) * 100
                print(f"    {pos.symbol}: {float(pos.qty):.4f} shares @ ${float(pos.avg_entry_price):.2f}  |  P&L: ${pnl:+.2f} ({pct:+.2f}%)")
        else:
            print("  No open positions")

        # Market data
        data_client = StockHistoricalDataClient(api_key, secret_key)
        end   = datetime.now()
        start = end - timedelta(days=365)

        print(f"\n  Market data test:")
        for ticker in TEST_TICKERS:
            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            ))
            df = bars.df
            if hasattr(df.index, 'levels'):
                df = df.xs(ticker, level='symbol')

            closes = df['close'].values.tolist()
            curr   = closes[-1]
            prev   = closes[-2]
            ma50   = np.mean(closes[-50:])
            date   = df.index[-1].date() if hasattr(df.index[-1], 'date') else str(df.index[-1])[:10]

            print(f"  {ticker}: ${curr:.2f}  ({((curr-prev)/prev)*100:+.2f}%)  |  MA50: ${ma50:.2f}  |  Date: {date}")

        print("\n✅ Alpaca data: working (real-time, no delay)")

    except ImportError:
        print("❌ alpaca-py not installed — run: pip install alpaca-py")
    except Exception as e:
        print(f"❌ Alpaca error: {e}")


# ============================================================
# COMPARE PRICES
# ============================================================
def compare_prices():
    print("\n" + "="*55)
    print("  PRICE COMPARISON (Yahoo vs Alpaca)")
    print("="*55)
    try:
        import yfinance as yf
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta

        api_key    = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key:
            print("Skipping — no Alpaca keys")
            return

        data_client = StockHistoricalDataClient(api_key, secret_key)
        end   = datetime.now()
        start = end - timedelta(days=5)

        print(f"\n  {'Ticker':<8} {'Yahoo':<12} {'Alpaca':<12} {'Diff'}")
        print(f"  {'-'*45}")

        for ticker in TEST_TICKERS:
            yf_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]

            bars = data_client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=ticker, timeframe=TimeFrame.Day, start=start, end=end
            ))
            df = bars.df
            if hasattr(df.index, 'levels'):
                df = df.xs(ticker, level='symbol')
            alp_price = df['close'].iloc[-1]

            diff = alp_price - yf_price
            flag = "✅" if abs(diff) < 0.10 else "⚠️ " if abs(diff) < 1.00 else "❌"
            print(f"  {ticker:<8} ${yf_price:<10.2f} ${alp_price:<10.2f} {flag} ${diff:+.2f}")

        print("\n  Small differences are normal (different data sources/timing).")
        print("  Large differences (>$1) would indicate a problem.")

    except Exception as e:
        print(f"  Could not compare: {e}")


if __name__ == "__main__":
    test_yfinance()
    test_alpaca()
    compare_prices()
    print("\n" + "="*55 + "\n")
