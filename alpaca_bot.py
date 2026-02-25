"""
alpaca_bot.py - Alpaca Paper Trading Integration for Meridian Markets
=====================================================================
Runs the SAME strategy as trading_bot.py but executes real orders
through Alpaca's paper trading API instead of simulating locally.

SETUP:
1. Sign up at alpaca.markets (free)
2. Switch to Paper Trading account in top-left of dashboard
3. Generate API keys from the Paper Trading dashboard
4. Add to your .env file:
   ALPACA_API_KEY=your_paper_api_key
   ALPACA_SECRET_KEY=your_paper_secret_key
   ALPACA_PAPER=true

pip install alpaca-py

DIFFERENCES FROM trading_bot.py:
- Prices come from Alpaca real-time feed (no 15-min delay)
- Orders submitted to Alpaca and filled at real market prices
- Stop losses are native broker orders — fire instantly, app doesn't need to be running
- Portfolio state read from Alpaca account, not our database
- Much more realistic simulation of real trading
"""

import os
from datetime import datetime, timedelta
from trading_bot import (
    compute_rsi, compute_macd, evaluate_buy_signal,
    get_ai_trade_reasoning, TRADEABLE_UNIVERSE,
    STARTING_CAPITAL, MAX_POSITIONS, MAX_POSITION_PCT,
    STOP_LOSS_PCT, TAKE_PROFIT_PCT
)


# ============================================================
# ALPACA CLIENTS
# ============================================================

def get_trading_client():
    try:
        from alpaca.trading.client import TradingClient
        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        paper  = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        if not key or not secret:
            raise ValueError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be in .env")
        return TradingClient(key, secret, paper=paper)
    except ImportError:
        raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")


def get_data_client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"),
        os.getenv("ALPACA_SECRET_KEY")
    )


# ============================================================
# REAL-TIME SIGNALS VIA ALPACA
# ============================================================

def get_alpaca_signals(ticker):
    """
    Fetch real-time data from Alpaca and compute indicators.
    Same logic as trading_bot.get_technical_signals() but uses
    Alpaca's real-time feed instead of yfinance (no 15-min delay).
    """
    try:
        import numpy as np
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        client  = get_data_client()
        end     = datetime.now()
        start   = end - timedelta(days=400)

        bars = client.get_stock_bars(StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        ))
        df = bars.df
        if df is None or len(df) < 60:
            return None

        if hasattr(df.index, 'levels'):
            df = df.xs(ticker, level='symbol')

        closes  = df['close'].values.tolist()
        volumes = df['volume'].values.tolist()
        curr, prev = closes[-1], closes[-2]

        ma50  = round(float(np.mean(closes[-50:])), 2)
        ma200 = round(float(np.mean(closes[-200:])), 2) if len(closes) >= 200 else None
        rsi   = compute_rsi(closes)

        macd, sig, _       = compute_macd(closes)
        macd_p, sig_p, _   = compute_macd(closes[:-1])

        vol_today = volumes[-1]
        vol_avg7  = float(np.mean(volumes[-8:-1])) if len(volumes) >= 8 else vol_today
        vol_ratio = round(vol_today / vol_avg7, 2) if vol_avg7 > 0 else 1.0

        return {
            "ticker":             ticker,
            "price":              round(float(curr), 2),
            "prev_price":         round(float(prev), 2),
            "change_pct":         round(((curr - prev) / prev) * 100, 2),
            "ma50":               ma50,
            "ma200":              ma200,
            "above_ma50":         curr > ma50,
            "above_ma200":        curr > ma200 if ma200 else None,
            "rsi":                rsi,
            "macd":               macd,
            "macd_signal":        sig,
            "macd_bullish_cross": macd_p is not None and macd_p < sig_p and macd > sig,
            "macd_bearish_cross": macd_p is not None and macd_p > sig_p and macd < sig,
            "volume":             int(vol_today),
            "vol_avg7":           int(vol_avg7),
            "vol_ratio":          vol_ratio,
            "high_vol":           vol_ratio > 1.1,
            "source":             "alpaca_realtime",
        }
    except Exception as e:
        print(f"Alpaca signal error {ticker}: {e}")
        return None


# ============================================================
# ORDER EXECUTION
# ============================================================

def submit_buy_order(ticker, notional, stop_price, target_price):
    """
    Submit bracket buy order — market buy with built-in stop loss and
    take profit. Bracket orders execute automatically at Alpaca even
    when our app is offline.
    Note: Bracket orders require whole share quantities (no fractional).
    We calculate qty from notional / current price and floor to integer.
    """
    try:
        from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        # Calculate whole share quantity — bracket orders don't support fractional
        qty = int(notional / stop_price * (1 / (1 - 0.05)))  # back-calc from stop
        # Simpler: just use notional / target_price floored to whole shares
        qty = max(1, int(notional / ((stop_price + target_price) / 2)))

        client = get_trading_client()
        order  = client.submit_order(MarketOrderRequest(
            symbol        = ticker,
            qty           = qty,
            side          = OrderSide.BUY,
            time_in_force = TimeInForce.DAY,
            order_class   = "bracket",
            take_profit   = TakeProfitRequest(limit_price=round(target_price, 2)),
            stop_loss     = StopLossRequest(stop_price=round(stop_price, 2)),
        ))
        return {"success": True, "order_id": str(order.id), "status": str(order.status), "qty": qty}
    except Exception as e:
        return {"success": False, "error": str(e)}


def submit_sell_order(ticker, reason="Signal"):
    """Close entire position via market order."""
    try:
        get_trading_client().close_position(ticker)
        return {"success": True, "ticker": ticker}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# PORTFOLIO STATE
# ============================================================

def get_alpaca_portfolio():
    """Read portfolio state from Alpaca account."""
    try:
        client   = get_trading_client()
        account  = client.get_account()
        positions = {}

        for pos in client.get_all_positions():
            positions[pos.symbol] = {
                "ticker":            pos.symbol,
                "shares":            float(pos.qty),
                "entry_price":       round(float(pos.avg_entry_price), 2),
                "current_price":     round(float(pos.current_price), 2),
                "market_value":      round(float(pos.market_value), 2),
                "unrealized_pnl":    round(float(pos.unrealized_pl), 2),
                "unrealized_pnl_pct": round(float(pos.unrealized_plpc) * 100, 2),
                "entry_date":        datetime.utcnow().isoformat(),
            }

        return {
            "source":          "alpaca",
            "cash":            round(float(account.cash), 2),
            "buying_power":    round(float(account.buying_power), 2),
            "portfolio_value": round(float(account.portfolio_value), 2),
            "starting_value":  STARTING_CAPITAL,
            "positions":       positions,
            "trade_history":   [],
            "equity_curve":    [],
            "created_at":      datetime.utcnow().isoformat(),
            "last_updated":    datetime.utcnow().isoformat(),
            "total_trades":    0,
            "winning_trades":  0,
        }
    except Exception as e:
        return {"error": str(e)}


def get_alpaca_stats():
    """Compute stats from live Alpaca account data."""
    try:
        from trading_bot import get_sp500_return
        client   = get_trading_client()
        account  = client.get_account()

        port_val     = float(account.portfolio_value)
        cash         = float(account.cash)
        total_return = round(((port_val - STARTING_CAPITAL) / STARTING_CAPITAL) * 100, 2)

        # S&P 500 benchmark — safe fallback
        try:
            sp500_return = get_sp500_return(
                (datetime.utcnow() - timedelta(days=30)).isoformat()
            )
        except:
            sp500_return = 0.0

        # Equity curve — multiple fallback attempts
        equity_curve = []
        try:
            history = client.get_portfolio_history(period="1M", timeframe="1D")
            if history and history.timestamp:
                for ts, val in zip(history.timestamp, history.equity):
                    if val:
                        equity_curve.append({
                            "date":  datetime.fromtimestamp(ts).strftime("%Y-%m-%d"),
                            "value": round(float(val), 2),
                        })
        except Exception as hist_err:
            print(f"Portfolio history unavailable: {hist_err}")
            # Seed with starting capital so chart has something to show
            equity_curve = [{"date": datetime.utcnow().strftime("%Y-%m-%d"), "value": round(port_val, 2)}]

        positions = client.get_all_positions()

        # Build unrealized P&L per position for chart
        positions_data = []
        for pos in positions:
            positions_data.append({
                "ticker":     pos.symbol,
                "pnl":        round(float(pos.unrealized_pl), 2),
                "pnl_pct":    round(float(pos.unrealized_plpc) * 100, 2),
                "market_val": round(float(pos.market_value), 2),
            })

        return {
            "current_value":   round(port_val, 2),
            "start_value":     STARTING_CAPITAL,
            "total_return":    total_return,
            "total_return_$":  round(port_val - STARTING_CAPITAL, 2),
            "sp500_return":    sp500_return,
            "alpha":           round(total_return - sp500_return, 2),
            "cash":            round(cash, 2),
            "buying_power":    round(float(account.buying_power), 2),
            "positions_value": round(port_val - cash, 2),
            "open_positions":  len(positions),
            "beat_market":     total_return > sp500_return,
            "equity_curve":    equity_curve,
            "positions_data":  positions_data,
            "source":          "alpaca",
            "win_rate":        0,
            "total_trades":    0,
            "winning_trades":  0,
            "total_pnl":       round(port_val - STARTING_CAPITAL, 2),
            "avg_win":         0,
            "avg_loss":        0,
            "max_drawdown":    0,
        }
    except Exception as e:
        print(f"get_alpaca_stats error: {e}")
        return {"error": str(e)}


# ============================================================
# MAIN BOT CYCLE
# ============================================================

def run_alpaca_cycle():
    """
    Full bot cycle using Alpaca for data and execution.
    Checks MACD/RSI exit signals on open positions, then
    scans universe for new buy signals.
    Note: Stop loss / take profit handled by bracket orders automatically.
    """
    actions  = []
    client   = get_trading_client()
    account  = client.get_account()
    held     = {p.symbol for p in client.get_all_positions()}
    positions = {p.symbol: p for p in client.get_all_positions()}

    # ── EXITS: Check MACD/RSI signals on open positions ──
    for ticker, pos in positions.items():
        signals = get_alpaca_signals(ticker)
        if not signals:
            continue

        curr       = float(pos.current_price)
        entry      = float(pos.avg_entry_price)
        pct_change = ((curr - entry) / entry) * 100

        should_sell = False
        sell_reason = ""

        if signals.get("macd_bearish_cross") and pct_change > 0:
            should_sell = True
            sell_reason = f"MACD bearish crossover at ${curr:.2f} (+{pct_change:.1f}%)"
        elif signals.get("rsi") and signals["rsi"] > 75 and pct_change > 5:
            should_sell = True
            sell_reason = f"RSI {signals['rsi']} overbought with {pct_change:.1f}% gain"

        if should_sell:
            result = submit_sell_order(ticker, sell_reason)
            if result["success"]:
                pnl = float(pos.unrealized_pl)
                ai  = get_ai_trade_reasoning("SELL", ticker, signals, [sell_reason],
                                             f"Alpaca paper — {len(positions)} positions open")
                actions.append({
                    "type": "SELL", "ticker": ticker, "price": curr,
                    "pnl": round(pnl, 2), "pnl_pct": round(pct_change, 2),
                    "reason": sell_reason, "ai_reasoning": ai, "source": "alpaca"
                })
                held.discard(ticker)

    # ── BUYS: Scan for new signals ──
    open_count   = len(client.get_all_positions())
    buying_power = float(account.buying_power)
    port_val     = float(account.portfolio_value)

    print(f"[Alpaca] Open positions: {open_count}/{MAX_POSITIONS} | Buying power: ${buying_power:,.2f}")

    if open_count >= MAX_POSITIONS:
        print(f"[Alpaca] All {MAX_POSITIONS} slots full — skipping buy scan")
        return actions

    print(f"[Alpaca] Scanning {len(TRADEABLE_UNIVERSE)} tickers for buy signals...")
    candidates = []
    for ticker in TRADEABLE_UNIVERSE:
        if ticker in held:
            print(f"[Alpaca]   {ticker} — already held, skipping")
            continue
        signals = get_alpaca_signals(ticker)
        if not signals:
            print(f"[Alpaca]   {ticker} — no signal data")
            continue
        should_buy, reasons, score = evaluate_buy_signal(signals)
        print(f"[Alpaca]   {ticker} — score:{score} buy:{should_buy} RSI:{signals.get('rsi','?')} MACD_cross:{signals.get('macd_bullish_cross','?')}")
        if should_buy:
            candidates.append((score, ticker, signals, reasons))

    print(f"[Alpaca] Found {len(candidates)} buy candidates")
    candidates.sort(key=lambda x: x[0], reverse=True)

    for score, ticker, signals, reasons in candidates[:MAX_POSITIONS - open_count]:
        curr_price   = signals["price"]
        notional     = min(port_val * MAX_POSITION_PCT, buying_power * 0.95)
        stop_price   = round(curr_price * (1 - STOP_LOSS_PCT), 2)
        target_price = round(curr_price * (1 + TAKE_PROFIT_PCT), 2)

        print(f"[Alpaca] Attempting BUY {ticker} @ ${curr_price} | notional:${notional:.0f} | stop:${stop_price} | target:${target_price}")

        if notional < 1 or buying_power < notional:
            print(f"[Alpaca]   Skipping {ticker} — insufficient buying power")
            continue

        result = submit_buy_order(ticker, notional, stop_price, target_price)
        if result["success"]:
            print(f"[Alpaca] ✅ BUY {ticker} x{result.get('qty','?')} shares @ ${curr_price} — ID:{result.get('order_id')}")
            ai = get_ai_trade_reasoning("BUY", ticker, signals, reasons,
                                        f"Alpaca paper — ${buying_power:,.0f} buying power")
            actions.append({
                "type": "BUY", "ticker": ticker, "price": curr_price,
                "notional": round(notional, 2), "stop": stop_price,
                "target": target_price, "score": score, "reasons": reasons,
                "ai_reasoning": ai, "order_id": result.get("order_id"), "source": "alpaca"
            })
            open_count   += 1
            buying_power -= notional
        else:
            print(f"[Alpaca] ❌ Order failed {ticker}: {result.get('error')}")

        if open_count >= MAX_POSITIONS:
            break

    print(f"[Alpaca] Cycle complete — {len(actions)} actions taken")
    return actions


def check_alpaca_connected():
    """Quick connectivity check. Returns (bool, message)."""
    try:
        client  = get_trading_client()
        account = client.get_account()
        return True, f"Connected — ${float(account.portfolio_value):,.2f} portfolio value"
    except Exception as e:
        return False, str(e)