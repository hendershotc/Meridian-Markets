"""
trading_bot.py - AI Paper Trading Simulator for Meridian Markets
================================================================
Designed for university students to run over a semester and compare
AI-driven trading performance vs. the S&P 500 benchmark.

TRADING STRATEGY:
The bot uses a multi-signal confirmation approach based on research:
  1. Trend Filter    — Price above/below 200-day MA (avoid trading against trend)
  2. Momentum Entry  — MACD crossover above signal line (momentum shift confirmed)
  3. RSI Conditions  — RSI between 35-65 on entry (not overbought/oversold)
  4. Volume Confirm  — Volume above 7-day average (institutional participation)
  5. MA Alignment    — Price above 50-day MA for longs (short-term trend intact)

EXIT RULES:
  - Stop loss:      5% below entry price (hard risk limit)
  - Take profit:    15% above entry price (2:1+ reward/risk target)
  - Trend exit:     MACD crosses below signal line (momentum reversing)
  - Trailing stop:  Once up 8%, stop moves to break even

POSITION SIZING:
  - Max 10% of portfolio per position
  - Max 5 open positions at once (diversification)
  - Tracks: cash, positions, trade history, P&L vs S&P 500

AI LAYER:
  - Claude analyzes each BUY/SELL decision with reasoning
  - Explains WHY the signal fired, what it means, what to watch
  - Educational context for students learning to trade
"""

import yfinance as yf
import numpy as np
import json
import os
from datetime import datetime, timedelta
from anthropic import Anthropic

# ── UNIVERSE OF TRADEABLE STOCKS ──
# Curated list of liquid, well-known stocks suitable for a semester simulator

TRADEABLE_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "CRM", "ORCL", "PLTR", "ADBE", "INTC", "QCOM",
    # Financials
    "JPM", "BAC", "GS", "V", "MA",
    "AXP", "MS", "BLK", "SCHW", "WFC",
    # Healthcare
    "JNJ", "LLY", "UNH", "PFE",
    "ABBV", "MRK", "TMO", "ABT", "REGN",
    # Consumer Discretionary
    "WMT", "COST", "MCD", "NKE",
    "AMZN", "HD", "SBUX", "TGT", "BKNG",
    # Consumer Staples
    "PG", "KO", "PEP", "PM",
    # Energy
    "XOM", "CVX",
    "COP", "SLB", "OXY", "MPC",
    # Industrial
    "CAT", "BA", "GE",
    "HON", "UPS", "LMT", "RTX", "DE",
    # Communication
    "NFLX", "DIS", "CMCSA", "T",
    # Materials & Real Estate
    "LIN", "FCX", "AMT", "PLD",
    # ETFs
    "SPY", "QQQ", "XLK", "XLF", "XLE",
    "XLV", "XLI", "XLP", "XLY", "GLD",
]

STARTING_CAPITAL  = 100_000.0   # $100,000 paper portfolio
MAX_POSITIONS     = 5           # max simultaneous holdings
MAX_POSITION_PCT  = 0.10        # max 10% per position
STOP_LOSS_PCT     = 0.05        # 5% stop loss
TAKE_PROFIT_PCT   = 0.15        # 15% take profit
BREAKEVEN_TRIGGER = 0.08        # move stop to breakeven once up 8%


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def compute_rsi(prices, period=14):
    """Compute RSI from price series."""
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        return 100.0
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    rs  = avg_gain / avg_loss if avg_loss != 0 else 100
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)


def compute_macd(prices, fast=12, slow=26, signal=9):
    """Compute MACD line, signal line, and histogram."""
    if len(prices) < slow + signal:
        return None, None, None
    prices  = np.array(prices, dtype=float)

    def ema(data, n):
        k = 2 / (n + 1)
        e = [data[0]]
        for p in data[1:]:
            e.append(p * k + e[-1] * (1 - k))
        return np.array(e)

    fast_ema  = ema(prices, fast)
    slow_ema  = ema(prices, slow)
    macd_line = fast_ema - slow_ema
    sig_line  = ema(macd_line, signal)
    histogram = macd_line - sig_line
    return (round(macd_line[-1], 4),
            round(sig_line[-1], 4),
            round(histogram[-1], 4))


def get_technical_signals(ticker):
    """
    Fetch price history and compute all signals for the bot.
    Returns a dict of all technical data needed for trade decisions.
    """
    try:
        stock   = yf.Ticker(ticker)
        hist    = stock.history(period="1y")
        if len(hist) < 60:
            return None

        closes  = hist["Close"].values.tolist()
        volumes = hist["Volume"].values.tolist()
        curr    = closes[-1]
        prev    = closes[-2]

        # Moving averages
        ma50  = round(np.mean(closes[-50:]), 2)
        ma200 = round(np.mean(closes[-200:]), 2) if len(closes) >= 200 else None

        # RSI
        rsi = compute_rsi(closes)

        # MACD (need previous bar too for crossover detection)
        macd, signal, hist_val = compute_macd(closes)
        macd_prev, signal_prev, _ = compute_macd(closes[:-1])

        # Volume
        vol_today = volumes[-1]
        vol_avg7  = np.mean(volumes[-8:-1]) if len(volumes) >= 8 else vol_today
        vol_ratio = round(vol_today / vol_avg7, 2) if vol_avg7 > 0 else 1.0

        # MACD crossover detection
        macd_bullish_cross = (macd_prev is not None and
                              macd_prev < signal_prev and
                              macd > signal)
        macd_bearish_cross = (macd_prev is not None and
                              macd_prev > signal_prev and
                              macd < signal)

        return {
            "ticker":             ticker,
            "price":              round(curr, 2),
            "prev_price":         round(prev, 2),
            "change_pct":         round(((curr - prev) / prev) * 100, 2),
            "ma50":               ma50,
            "ma200":              ma200,
            "above_ma50":         curr > ma50,
            "above_ma200":        curr > ma200 if ma200 else None,
            "rsi":                rsi,
            "macd":               macd,
            "macd_signal":        signal,
            "macd_hist":          hist_val,
            "macd_bullish_cross": macd_bullish_cross,
            "macd_bearish_cross": macd_bearish_cross,
            "volume":             int(vol_today),
            "vol_avg7":           int(vol_avg7),
            "vol_ratio":          vol_ratio,
            "high_vol":           vol_ratio > 1.1,
        }
    except Exception as e:
        return None


# ============================================================
# BUY / SELL SIGNAL LOGIC
# ============================================================

def evaluate_buy_signal(signals):
    """
    Multi-condition buy signal.
    Returns (should_buy: bool, reasons: list, score: int)
    Score 0-5 based on how many conditions are met.
    Requires at least 3/5 conditions for a buy.
    """
    if not signals:
        return False, [], 0

    conditions = []
    reasons    = []

    # 1. Trend filter — price above 200 MA
    if signals.get("above_ma200"):
        conditions.append(True)
        reasons.append(f"Price ${signals['price']} above 200MA ${signals['ma200']} (uptrend)")
    else:
        conditions.append(False)
        reasons.append(f"Price below 200MA — trend caution")

    # 2. Short-term trend — price above 50 MA
    if signals.get("above_ma50"):
        conditions.append(True)
        reasons.append(f"Price above 50MA ${signals['ma50']} (short-term bullish)")
    else:
        conditions.append(False)
        reasons.append(f"Price below 50MA — short-term weakness")

    # 3. MACD bullish crossover
    if signals.get("macd_bullish_cross"):
        conditions.append(True)
        reasons.append(f"MACD crossed above signal line (momentum shift bullish)")
    elif signals.get("macd") and signals.get("macd_signal") and signals["macd"] > signals["macd_signal"]:
        conditions.append(True)
        reasons.append(f"MACD above signal line (bullish momentum active)")
    else:
        conditions.append(False)
        reasons.append(f"MACD below signal line (bearish momentum)")

    # 4. RSI not overbought — sweet spot 35-65
    rsi = signals.get("rsi")
    if rsi and 35 <= rsi <= 65:
        conditions.append(True)
        reasons.append(f"RSI {rsi} in ideal range 35-65 (not overbought)")
    elif rsi and rsi < 35:
        conditions.append(True)
        reasons.append(f"RSI {rsi} oversold — potential reversal entry")
    else:
        conditions.append(False)
        reasons.append(f"RSI {rsi} overbought (>65) — risky entry")

    # 5. Volume confirmation
    if signals.get("high_vol"):
        conditions.append(True)
        reasons.append(f"Volume {signals['vol_ratio']}x average (institutional confirmation)")
    else:
        conditions.append(False)
        reasons.append(f"Below average volume ({signals['vol_ratio']}x) — weak confirmation")

    score      = sum(conditions)
    should_buy = score >= 3

    return should_buy, reasons, score


def evaluate_sell_signal(signals, position):
    """
    Check if an open position should be closed.
    Returns (should_sell: bool, reason: str)
    """
    if not signals or not position:
        return False, ""

    entry_price    = position["entry_price"]
    current_price  = signals["price"]
    pct_change     = ((current_price - entry_price) / entry_price) * 100
    stop_price     = position.get("stop_price", entry_price * (1 - STOP_LOSS_PCT))
    target_price   = entry_price * (1 + TAKE_PROFIT_PCT)

    # Hard stop loss
    if current_price <= stop_price:
        return True, f"Stop loss hit at ${current_price:.2f} (entry ${entry_price:.2f}, loss {pct_change:.1f}%)"

    # Take profit target
    if current_price >= target_price:
        return True, f"Take profit target hit at ${current_price:.2f} (+{pct_change:.1f}%)"

    # MACD bearish crossover — momentum reversing
    if signals.get("macd_bearish_cross"):
        return True, f"MACD bearish crossover — momentum reversing at ${current_price:.2f} ({pct_change:+.1f}%)"

    # RSI overbought while in profit
    rsi = signals.get("rsi")
    if rsi and rsi > 75 and pct_change > 5:
        return True, f"RSI {rsi} overbought with {pct_change:.1f}% gain — taking profits"

    return False, ""


# ============================================================
# AI REASONING ENGINE
# ============================================================

def get_ai_trade_reasoning(action, ticker, signals, reasons, portfolio_context):
    """
    Get Claude's educational explanation of the trade decision.
    Designed for university students learning trading.
    """
    try:
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        signals_str = json.dumps({k: v for k, v in signals.items()
                                  if k in ["price", "rsi", "macd", "macd_signal",
                                           "ma50", "ma200", "vol_ratio",
                                           "macd_bullish_cross", "macd_bearish_cross",
                                           "above_ma50", "above_ma200", "change_pct"]},
                                 indent=2)

        reasons_str = "\n".join(f"- {r}" for r in reasons)

        if action == "BUY":
            prompt = f"""You are an AI trading bot explaining a BUY decision to a university finance student.

TICKER: {ticker}
TECHNICAL SIGNALS:
{signals_str}

CONDITIONS MET:
{reasons_str}

PORTFOLIO: {portfolio_context}

Explain this BUY decision in 3 short sections:
REASONING: Why the technical signals support buying right now. Reference the specific indicators.
WHAT TO WATCH: 2 key things the student should monitor after this trade is entered.
EDUCATIONAL NOTE: One concise lesson about the indicator combination that triggered this trade.

Keep each section 2-3 sentences. Plain text, no markdown."""

        else:  # SELL
            prompt = f"""You are an AI trading bot explaining a SELL/EXIT decision to a university finance student.

TICKER: {ticker}
EXIT REASON: {reasons[0] if reasons else 'Signal triggered'}
TECHNICAL SIGNALS:
{signals_str}

PORTFOLIO: {portfolio_context}

Explain this exit decision in 3 short sections:
REASONING: Why the bot is exiting this position now.
OUTCOME: What this trade result teaches about the strategy.
EDUCATIONAL NOTE: One concise lesson about exit discipline and risk management.

Keep each section 2-3 sentences. Plain text, no markdown."""

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"AI reasoning unavailable: {str(e)}"


# ============================================================
# PORTFOLIO STATE MANAGEMENT
# ============================================================

def create_new_portfolio(user_id):
    """Create a fresh paper trading portfolio."""
    return {
        "user_id":        user_id,
        "cash":           STARTING_CAPITAL,
        "starting_value": STARTING_CAPITAL,
        "positions":      {},       # ticker -> position dict
        "trade_history":  [],       # list of completed trades
        "equity_curve":   [{"date": datetime.utcnow().strftime("%Y-%m-%d"), "value": STARTING_CAPITAL, "sp500": 100.0}],
        "pending_signals":[],       # signals evaluated but not acted on
        "created_at":     datetime.utcnow().isoformat(),
        "last_updated":   datetime.utcnow().isoformat(),
        "is_active":      True,
        "total_trades":   0,
        "winning_trades": 0,
    }


def get_portfolio_value(portfolio):
    """Calculate current total portfolio value including open positions."""
    total = portfolio["cash"]
    for ticker, pos in portfolio["positions"].items():
        try:
            signals = get_technical_signals(ticker)
            if signals:
                total += signals["price"] * pos["shares"]
        except:
            total += pos["entry_price"] * pos["shares"]
    return round(total, 2)


def get_sp500_return(start_date_str):
    """Get S&P 500 return since portfolio start date for benchmarking."""
    try:
        start  = datetime.fromisoformat(start_date_str)
        spy    = yf.Ticker("SPY")
        hist   = spy.history(start=start.strftime("%Y-%m-%d"))
        if len(hist) < 2:
            return 0.0
        start_price = hist["Close"].iloc[0]
        curr_price  = hist["Close"].iloc[-1]
        return round(((curr_price - start_price) / start_price) * 100, 2)
    except:
        return 0.0


def run_bot_cycle(portfolio):
    """
    Run one full cycle of the trading bot:
    1. Check all open positions for exit signals
    2. Scan universe for new buy signals
    3. Execute trades and log reasoning
    Returns updated portfolio and list of actions taken.
    """
    actions = []

    # ── STEP 1: CHECK EXITS ON OPEN POSITIONS ──
    for ticker in list(portfolio["positions"].keys()):
        signals  = get_technical_signals(ticker)
        position = portfolio["positions"][ticker]
        if not signals:
            continue

        # Update trailing stop — once up 8%, stop moves to breakeven
        entry = position["entry_price"]
        curr  = signals["price"]
        pct   = ((curr - entry) / entry) * 100
        if pct >= BREAKEVEN_TRIGGER * 100:
            new_stop = max(position.get("stop_price", entry * (1 - STOP_LOSS_PCT)), entry)
            portfolio["positions"][ticker]["stop_price"] = new_stop

        should_sell, sell_reason = evaluate_sell_signal(signals, position)

        if should_sell:
            # Execute sell
            shares       = position["shares"]
            proceeds     = shares * curr
            entry_value  = shares * entry
            pnl          = proceeds - entry_value
            pnl_pct      = round(((curr - entry) / entry) * 100, 2)

            # Get AI reasoning
            portfolio_ctx = f"Portfolio value ~${get_portfolio_value(portfolio):,.0f}, {len(portfolio['positions'])} open positions"
            ai_reasoning  = get_ai_trade_reasoning("SELL", ticker, signals, [sell_reason], portfolio_ctx)

            # Record trade
            trade = {
                "ticker":       ticker,
                "action":       "SELL",
                "shares":       shares,
                "entry_price":  entry,
                "exit_price":   round(curr, 2),
                "pnl":          round(pnl, 2),
                "pnl_pct":      pnl_pct,
                "reason":       sell_reason,
                "ai_reasoning": ai_reasoning,
                "entry_date":   position["entry_date"],
                "exit_date":    datetime.utcnow().isoformat(),
                "winner":       pnl > 0,
            }
            portfolio["trade_history"].append(trade)
            portfolio["cash"] += proceeds
            portfolio["total_trades"] += 1
            if pnl > 0:
                portfolio["winning_trades"] += 1
            del portfolio["positions"][ticker]

            actions.append({
                "type":   "SELL",
                "ticker": ticker,
                "price":  curr,
                "pnl":    round(pnl, 2),
                "pnl_pct": pnl_pct,
                "reason": sell_reason,
                "ai_reasoning": ai_reasoning,
            })

    # ── STEP 2: SCAN FOR NEW BUY SIGNALS ──
    open_count = len(portfolio["positions"])
    if open_count >= MAX_POSITIONS:
        return portfolio, actions  # portfolio is full

    # Don't buy the same tickers already held
    held        = set(portfolio["positions"].keys())
    candidates  = [t for t in TRADEABLE_UNIVERSE if t not in held]

    buy_candidates = []
    for ticker in candidates:
        signals            = get_technical_signals(ticker)
        should_buy, reasons, score = evaluate_buy_signal(signals)
        if should_buy:
            buy_candidates.append((score, ticker, signals, reasons))

    # Sort by score descending — best signals first
    buy_candidates.sort(key=lambda x: x[0], reverse=True)

    slots_available = MAX_POSITIONS - open_count
    for score, ticker, signals, reasons in buy_candidates[:slots_available]:
        curr_price    = signals["price"]
        max_position  = portfolio["cash"] * MAX_POSITION_PCT
        shares        = int(max_position / curr_price)

        if shares < 1 or portfolio["cash"] < curr_price:
            continue

        cost          = shares * curr_price
        stop_price    = round(curr_price * (1 - STOP_LOSS_PCT), 2)
        target_price  = round(curr_price * (1 + TAKE_PROFIT_PCT), 2)

        # Get AI reasoning
        portfolio_ctx = f"Portfolio value ~${get_portfolio_value(portfolio):,.0f}, {open_count} open positions, ${portfolio['cash']:,.0f} cash"
        ai_reasoning  = get_ai_trade_reasoning("BUY", ticker, signals, reasons, portfolio_ctx)

        # Execute buy
        position = {
            "ticker":       ticker,
            "shares":       shares,
            "entry_price":  round(curr_price, 2),
            "stop_price":   stop_price,
            "target_price": target_price,
            "entry_date":   datetime.utcnow().isoformat(),
            "signal_score": score,
            "entry_reasons": reasons,
            "ai_reasoning": ai_reasoning,
        }
        portfolio["positions"][ticker] = position
        portfolio["cash"] -= cost
        open_count        += 1

        actions.append({
            "type":         "BUY",
            "ticker":       ticker,
            "price":        round(curr_price, 2),
            "shares":       shares,
            "cost":         round(cost, 2),
            "stop":         stop_price,
            "target":       target_price,
            "score":        score,
            "reasons":      reasons,
            "ai_reasoning": ai_reasoning,
        })

        if open_count >= MAX_POSITIONS:
            break

    portfolio["last_updated"] = datetime.utcnow().isoformat()

    # Snapshot equity curve daily
    today         = datetime.utcnow().strftime("%Y-%m-%d")
    current_value = get_portfolio_value(portfolio)
    curve         = portfolio.get("equity_curve", [])
    sp500_return  = get_sp500_return(portfolio["created_at"])
    sp500_indexed = round(100 * (1 + sp500_return / 100), 2)
    if not curve or curve[-1]["date"] != today:
        curve.append({"date": today, "value": current_value, "sp500": sp500_indexed})
        portfolio["equity_curve"] = curve

    return portfolio, actions


def get_portfolio_stats(portfolio):
    """Compute comprehensive portfolio statistics."""
    current_value  = get_portfolio_value(portfolio)
    start_value    = portfolio["starting_value"]
    total_return   = round(((current_value - start_value) / start_value) * 100, 2)
    sp500_return   = get_sp500_return(portfolio["created_at"])
    alpha          = round(total_return - sp500_return, 2)

    trades         = portfolio["trade_history"]
    total_trades   = len(trades)
    winning_trades = sum(1 for t in trades if t.get("winner"))
    win_rate       = round((winning_trades / total_trades * 100), 1) if total_trades > 0 else 0
    total_pnl      = round(sum(t.get("pnl", 0) for t in trades), 2)

    avg_win        = 0
    avg_loss       = 0
    winners        = [t["pnl"] for t in trades if t.get("winner")]
    losers         = [t["pnl"] for t in trades if not t.get("winner") and t.get("pnl") is not None]
    if winners:
        avg_win    = round(np.mean(winners), 2)
    if losers:
        avg_loss   = round(np.mean(losers), 2)

    # Max drawdown from trade history
    running_pnl = 0
    peak        = 0
    max_dd      = 0
    for t in trades:
        running_pnl += t.get("pnl", 0)
        if running_pnl > peak:
            peak = running_pnl
        dd = peak - running_pnl
        if dd > max_dd:
            max_dd = dd

    return {
        "current_value":   current_value,
        "start_value":     start_value,
        "total_return":    total_return,
        "total_return_$":  round(current_value - start_value, 2),
        "sp500_return":    sp500_return,
        "alpha":           alpha,
        "cash":            round(portfolio["cash"], 2),
        "positions_value": round(current_value - portfolio["cash"], 2),
        "open_positions":  len(portfolio["positions"]),
        "total_trades":    total_trades,
        "winning_trades":  winning_trades,
        "win_rate":        win_rate,
        "total_pnl":       total_pnl,
        "avg_win":         avg_win,
        "avg_loss":        avg_loss,
        "max_drawdown":    round(max_dd, 2),
        "beat_market":     total_return > sp500_return,
    }


# ============================================================
# INTRADAY STOP LOSS CHECK (no new buys, exits only)
# ============================================================

def run_intraday_check(portfolio):
    """
    Lightweight check for stop loss / take profit on open positions only.
    Does NOT open new positions. Used for 9:45 AM and 12:30 PM checks.
    """
    actions = []
    for ticker in list(portfolio["positions"].keys()):
        signals  = get_technical_signals(ticker)
        position = portfolio["positions"][ticker]
        if not signals:
            continue

        # Update trailing stop
        entry = position["entry_price"]
        curr  = signals["price"]
        pct   = ((curr - entry) / entry) * 100
        if pct >= BREAKEVEN_TRIGGER * 100:
            new_stop = max(position.get("stop_price", entry * (1 - STOP_LOSS_PCT)), entry)
            portfolio["positions"][ticker]["stop_price"] = new_stop

        should_sell, sell_reason = evaluate_sell_signal(signals, position)
        if should_sell:
            shares      = position["shares"]
            proceeds    = shares * curr
            entry_value = shares * entry
            pnl         = proceeds - entry_value
            pnl_pct     = round(((curr - entry) / entry) * 100, 2)

            trade = {
                "ticker":       ticker,
                "action":       "SELL",
                "shares":       shares,
                "entry_price":  entry,
                "exit_price":   round(curr, 2),
                "pnl":          round(pnl, 2),
                "pnl_pct":      pnl_pct,
                "reason":       f"[INTRADAY] {sell_reason}",
                "ai_reasoning": "",
                "entry_date":   position["entry_date"],
                "exit_date":    datetime.utcnow().isoformat(),
                "winner":       pnl > 0,
            }
            portfolio["trade_history"].append(trade)
            portfolio["cash"] += proceeds
            portfolio["total_trades"] += 1
            if pnl > 0:
                portfolio["winning_trades"] += 1
            del portfolio["positions"][ticker]
            actions.append({"type": "SELL", "ticker": ticker, "price": curr,
                             "pnl": round(pnl, 2), "pnl_pct": pnl_pct, "reason": sell_reason})

    portfolio["last_updated"] = datetime.utcnow().isoformat()
    return portfolio, actions


# ============================================================
# SCHEDULER — runs automatically 3x per day
# ============================================================

def init_scheduler(app, db, SimulatorPortfolio):
    """
    Initialize APScheduler with 3 daily jobs:
      9:45 AM  EST — intraday stop check (exits only)
      12:30 PM EST — midday stop check (exits only)
      4:15 PM  EST — end of day full cycle (exits + new buys)
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.cron import CronTrigger
        import pytz

        eastern = pytz.timezone("America/New_York")
        scheduler = BackgroundScheduler(timezone=eastern)

        def run_for_all_users(mode="full"):
            """Run bot cycle for every active portfolio."""
            with app.app_context():
                active = SimulatorPortfolio.query.filter_by(is_active=True).all()
                for row in active:
                    try:
                        portfolio = json.loads(row.portfolio_json)
                        if mode == "intraday":
                            portfolio, _ = run_intraday_check(portfolio)
                        else:
                            portfolio, _ = run_bot_cycle(portfolio)
                        row.portfolio_json = json.dumps(portfolio)
                        row.updated_at     = datetime.utcnow()
                    except Exception as e:
                        print(f"Scheduler error for portfolio {row.id}: {e}")
                db.session.commit()
                print(f"Scheduler {mode} run complete for {len(active)} portfolios")

        # 9:45 AM EST — intraday check
        scheduler.add_job(
            lambda: run_for_all_users("intraday"),
            CronTrigger(hour=9, minute=45, day_of_week="mon-fri", timezone=eastern),
            id="morning_check"
        )

        # 12:30 PM EST — midday check
        scheduler.add_job(
            lambda: run_for_all_users("intraday"),
            CronTrigger(hour=12, minute=30, day_of_week="mon-fri", timezone=eastern),
            id="midday_check"
        )

        # 4:15 PM EST — end of day full cycle
        scheduler.add_job(
            lambda: run_for_all_users("full"),
            CronTrigger(hour=16, minute=15, day_of_week="mon-fri", timezone=eastern),
            id="eod_full"
        )

        # ── ALPACA PAPER TRADING JOBS ──
        def run_alpaca_scheduled(mode="full"):
            """Run Alpaca bot cycle if API keys are configured."""
            try:
                from alpaca_bot import run_alpaca_cycle, check_alpaca_connected
                connected, _ = check_alpaca_connected()
                if not connected:
                    return
                if mode == "full":
                    with app.app_context():
                        actions = run_alpaca_cycle()
                        print(f"Alpaca scheduler ({mode}): {len(actions)} actions taken")
            except Exception as e:
                print(f"Alpaca scheduler error: {e}")

        scheduler.add_job(
            lambda: run_alpaca_scheduled("intraday"),
            CronTrigger(hour=9, minute=45, day_of_week="mon-fri", timezone=eastern),
            id="alpaca_morning"
        )
        scheduler.add_job(
            lambda: run_alpaca_scheduled("intraday"),
            CronTrigger(hour=12, minute=30, day_of_week="mon-fri", timezone=eastern),
            id="alpaca_midday"
        )
        scheduler.add_job(
            lambda: run_alpaca_scheduled("full"),
            CronTrigger(hour=16, minute=15, day_of_week="mon-fri", timezone=eastern),
            id="alpaca_eod"
        )

        scheduler.start()
        print("Trading bot scheduler started — running Mon-Fri at 9:45, 12:30, 16:15 EST")
        print("Alpaca scheduler also active — will run if API keys are configured")
        return scheduler

    except ImportError:
        print("APScheduler not installed — run: pip install apscheduler pytz")
        return None