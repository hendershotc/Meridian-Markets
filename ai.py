"""
ai.py - AI analysis for Meridian Markets
Five focused functions:
  1. get_quick_snap      — 3-line watchlist signal (fast, uses Haiku)
  2. get_stock_deep_dive — full stock analysis for /stock/<ticker> page
  3. get_sector_analysis — sector drill-down panel
  4. get_macro_analysis  — broad market overview
  5. get_portfolio_analysis — full watchlist review
"""

import anthropic
import os


def _client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def _price_ctx(p):
    if not p or "error" in p:
        return "Price data unavailable"
    return (
        f"Price: ${p.get('price')} ({p.get('change_pct')}% today) | "
        f"52W: ${p.get('low_52w')} - ${p.get('high_52w')} | "
        f"50MA: ${p.get('ma50')} | 200MA: ${p.get('ma200')} | "
        f"MA Context: {p.get('ma_context')} | "
        f"Volume: {p.get('vol_signal')} ({p.get('vol_ratio')}x avg)"
    )


def _fund_ctx(f):
    if not f or "error" in f or "note" in f:
        return "Fundamentals unavailable"
    return (
        f"P/E: {f.get('pe_ratio')} | Fwd P/E: {f.get('forward_pe')} | EPS: {f.get('eps')} | "
        f"Rev Growth: {f.get('revenue_growth')} | Margin: {f.get('profit_margin')} | "
        f"ROE: {f.get('roe')} | MCap: {f.get('market_cap')} | D/E: {f.get('debt_to_equity')} | "
        f"Beta: {f.get('beta')} | Short%: {f.get('short_pct')} ({f.get('short_signal')}) | "
        f"Short Ratio: {f.get('short_ratio')}d | Inst Own: {f.get('inst_ownership')} | "
        f"Options Expiry: {f.get('next_expiry')}"
    )


def _parse(text, key_map):
    sections      = {k: "" for k in key_map}
    lines         = text.strip().split("\n")
    current_key   = None
    current_lines = []
    for line in lines:
        stripped = line.strip()
        matched  = False
        for key, labels in key_map.items():
            for label in labels:
                if stripped.upper().startswith(label):
                    if current_key:
                        sections[current_key] = " ".join(current_lines).strip()
                    current_key   = key
                    remainder     = stripped[len(label):].lstrip(":. ").strip()
                    current_lines = [remainder] if remainder else []
                    matched       = True
                    break
            if matched:
                break
        if not matched and current_key and stripped:
            current_lines.append(stripped)
    if current_key:
        sections[current_key] = " ".join(current_lines).strip()
    return sections


# ============================================================
# 1. QUICK SNAP — fast 3-line watchlist signal
# ============================================================
def get_quick_snap(ticker, price_data, fundamentals):
    try:
        prompt = f"""You are a professional trader. Rapid-fire assessment of {ticker}.

DATA: {_price_ctx(price_data)} | {_fund_ctx(fundamentals)}

Respond with exactly 3 labeled lines, one sentence each:
SIGNAL: BUY / HOLD / WATCH / AVOID and one reason.
EDGE: Single most important factor right now.
RISK: Biggest risk to the thesis.

No markdown. No extra text."""

        msg = _client().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse(msg.content[0].text, {
            "signal": ["SIGNAL"],
            "edge":   ["EDGE"],
            "risk":   ["RISK"],
        })
    except Exception as e:
        return {"signal": "Unavailable", "edge": "", "risk": str(e)}


# ============================================================
# 2. STOCK DEEP DIVE — full analysis for stock detail page
# ============================================================
def get_stock_deep_dive(ticker, price_data, fundamentals, news_headlines, earnings_date):
    news_str     = "\n".join([f"- {h['title']}" for h in news_headlines]) if news_headlines else "No recent news"
    earnings_str = f"{earnings_date['date']} ({earnings_date['label']}, {earnings_date['timing']})" if earnings_date else "None in next 2 weeks"

    try:
        prompt = f"""You are a senior equity analyst. Advanced trader audience. Analyze {ticker}.

PRICE & TECHNICALS: {_price_ctx(price_data)}
FUNDAMENTALS: {_fund_ctx(fundamentals)}
NEWS:
{news_str}
EARNINGS: {earnings_str}

Provide analysis with these labeled sections, 2-3 sentences each. Be direct and specific.

1. THESIS: Bull or bear thesis right now.
2. TECHNICALS: MA structure, volume, key levels.
3. FUNDAMENTALS: Growth, margins, valuation quality.
4. SIGNAL: BUY / HOLD / WATCH / AVOID with clear reasoning.
5. CATALYST: What could move this stock in the next 2-4 weeks.
6. TRADE SETUP: Entry zone, stop reference, upside target.
7. KEY RISKS: Top 2 risks that could invalidate the thesis.

Plain text only. No markdown. No asterisks."""

        msg = _client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        text     = msg.content[0].text
        sections = _parse(text, {
            "thesis":       ["1. THESIS", "THESIS"],
            "technicals":   ["2. TECHNICALS", "TECHNICALS"],
            "fundamentals": ["3. FUNDAMENTALS", "FUNDAMENTALS"],
            "signal":       ["4. SIGNAL", "SIGNAL"],
            "catalyst":     ["5. CATALYST", "CATALYST"],
            "trade_setup":  ["6. TRADE SETUP", "TRADE SETUP"],
            "risks":        ["7. KEY RISKS", "KEY RISKS"],
        })
        sig = sections.get("signal", "").upper()
        sections["signal_word"] = "BUY" if "BUY" in sig else "AVOID" if "AVOID" in sig else "WATCH" if "WATCH" in sig else "HOLD"
        return sections, msg.usage.input_tokens, msg.usage.output_tokens
    except Exception as e:
        return {"thesis": f"Analysis unavailable: {str(e)}"}, 0, 0


# ============================================================
# 3. SECTOR ANALYSIS — sector drill-down panel
# ============================================================
def get_sector_analysis(sector_name, sector_stocks, etf_change):
    stocks_str = "\n".join([
        f"- {s['ticker']} ({s['name']}): ${s['price']} ({'+' if s['change_pct'] >= 0 else ''}{s['change_pct']}%)"
        for s in sector_stocks
    ])
    try:
        prompt = f"""You are a sector analyst. Advanced trader audience.

SECTOR: {sector_name} | ETF TODAY: {'+' if etf_change >= 0 else ''}{etf_change}%
STOCKS:
{stocks_str}

Labeled sections, 2-3 sentences each:
1. OVERVIEW: Sector state and today's performance in context.
2. DRIVERS: Key factors driving this sector now.
3. LEADERS & LAGGARDS: What relative performance signals.
4. OUTLOOK: Near-term view and what to watch.
5. RISK FACTORS: Key risks facing this sector.

Plain text only. No markdown."""

        msg = _client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse(msg.content[0].text, {
            "overview": ["1. OVERVIEW", "OVERVIEW"],
            "drivers":  ["2. DRIVERS", "DRIVERS"],
            "leaders":  ["3. LEADERS & LAGGARDS", "LEADERS & LAGGARDS", "3. LEADERS"],
            "outlook":  ["4. OUTLOOK", "OUTLOOK"],
            "risks":    ["5. RISK FACTORS", "RISK FACTORS"],
        })
    except Exception as e:
        return {"overview": f"Unavailable: {str(e)}"}


# ============================================================
# 4. MACRO ANALYSIS — home page market overview
# ============================================================
def get_macro_analysis(macro_data, vix_level):
    macro_str   = "\n".join([
        f"- {m['label']} ({m['ticker']}): ${m['price']} ({'+' if m['change_pct'] >= 0 else ''}{m['change_pct']}%)"
        for m in macro_data
    ])
    vix_context = "elevated fear" if vix_level > 25 else "moderate uncertainty" if vix_level > 18 else "low volatility"

    try:
        prompt = f"""You are a macro strategist. Advanced trader audience.

MARKET DATA:
{macro_str}
VIX: {vix_level} ({vix_context})

Labeled sections, 2-3 sentences each:
1. MACRO OVERVIEW: Market conditions and cross-asset dynamics.
2. MARKET SENTIMENT: Risk-on or risk-off and what signals it.
3. SECTOR ROTATION: Where smart money is likely rotating today.
4. MICRO SIGNALS: What gold, oil, bonds, dollar tell us about macro.
5. INVESTOR PLAYBOOK: 3 specific actionable observations for today.
6. KEY RISKS: Top 3 macro risks to monitor this week.

Plain text only. No markdown."""

        msg = _client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse(msg.content[0].text, {
            "overview":  ["1. MACRO OVERVIEW", "MACRO OVERVIEW"],
            "sentiment": ["2. MARKET SENTIMENT", "MARKET SENTIMENT"],
            "rotation":  ["3. SECTOR ROTATION", "SECTOR ROTATION"],
            "micro":     ["4. MICRO SIGNALS", "MICRO SIGNALS"],
            "playbook":  ["5. INVESTOR PLAYBOOK", "INVESTOR PLAYBOOK"],
            "risks":     ["6. KEY RISKS", "KEY RISKS"],
        })
    except Exception as e:
        return {"overview": f"Unavailable: {str(e)}"}


# ============================================================
# 5. PORTFOLIO ANALYSIS — full watchlist review
# ============================================================
def get_portfolio_analysis(watchlist_data):
    stocks_str = "\n".join([
        f"- {s['ticker']}: ${s['price']} ({'+' if s['change_pct'] >= 0 else ''}{s['change_pct']}%) | "
        f"Sector: {s.get('sector','?')} | Signal: {s.get('signal','?')} | "
        f"MA: {s.get('ma_context','?')} | Vol: {s.get('vol_signal','?')}"
        for s in watchlist_data
    ])

    try:
        prompt = f"""You are a portfolio manager. Advanced trader audience.

WATCHLIST:
{stocks_str}

Labeled sections, 2-4 sentences each:
1. PORTFOLIO OVERVIEW: Overall health, bullish vs bearish breakdown.
2. SECTOR EXPOSURE: Concentration risks and diversification quality.
3. STRONGEST POSITIONS: Top 2-3 setups and why they stand out.
4. WEAKEST POSITIONS: 1-2 warning signs and what they signal.
5. PORTFOLIO RISK: Overall risk level based on current data.
6. RECOMMENDED ACTIONS: 3-4 direct, specific recommendations.

Plain text only. No markdown."""

        msg = _client().messages.create(
            model="claude-sonnet-4-6",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}]
        )
        return _parse(msg.content[0].text, {
            "overview":  ["1. PORTFOLIO OVERVIEW", "PORTFOLIO OVERVIEW"],
            "exposure":  ["2. SECTOR EXPOSURE", "SECTOR EXPOSURE"],
            "strongest": ["3. STRONGEST POSITIONS", "STRONGEST POSITIONS"],
            "weakest":   ["4. WEAKEST POSITIONS", "WEAKEST POSITIONS"],
            "risk":      ["5. PORTFOLIO RISK", "PORTFOLIO RISK"],
            "actions":   ["6. RECOMMENDED ACTIONS", "RECOMMENDED ACTIONS"],
        })
    except Exception as e:
        return {"overview": f"Unavailable: {str(e)}"}