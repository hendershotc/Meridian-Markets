"""
Merdian Markets - Main Application
A multi-user AI-powered market intelligence platform.

SETUP:
1. pip install flask flask-sqlalchemy flask-login yfinance anthropic stripe python-dotenv
2. Copy .env.example to .env and fill in your credentials
3. python app.py
"""

from flask import Flask, render_template, redirect, url_for, flash, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask.json.provider import DefaultJSONProvider
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import os

app = Flask(__name__)
app.config["SECRET_KEY"]           = os.getenv("SECRET_KEY", "dev-secret-change-in-production")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///marketpulse.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db           = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

class NumpySafeProvider(DefaultJSONProvider):
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.bool_):    return bool(obj)
            if isinstance(obj, np.integer):  return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray):  return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, bool): return bool(obj)
        return super().default(obj)
    
app.json_provider_class = NumpySafeProvider
app.json = NumpySafeProvider(app)

# ============================================================
# MODELS
# ============================================================
class User(UserMixin, db.Model):
    id               = db.Column(db.Integer, primary_key=True)
    email            = db.Column(db.String(150), unique=True, nullable=False)
    password_hash    = db.Column(db.String(256), nullable=False)
    name             = db.Column(db.String(100), nullable=False)
    experience_level = db.Column(db.String(20), default="beginner")   # beginner, intermediate, advanced
    is_pro           = db.Column(db.Boolean, default=False)            # unlocks AI features
    created_at       = db.Column(db.DateTime, default=datetime.utcnow)
    watchlist        = db.relationship("WatchlistItem", backref="user", lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class WatchlistItem(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    user_id    = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    ticker     = db.Column(db.String(10), nullable=False)
    added_at   = db.Column(db.DateTime, default=datetime.utcnow)


class DonationLog(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=True)
    amount_cents   = db.Column(db.Integer, nullable=False)
    stripe_session = db.Column(db.String(256), nullable=True)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)


class SimulatorPortfolio(db.Model):
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    portfolio_json = db.Column(db.Text, nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at     = db.Column(db.DateTime, default=datetime.utcnow)
    is_active      = db.Column(db.Boolean, default=True)
    user           = db.relationship("User", backref="simulators")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ============================================================
# ROUTES - Auth
# ============================================================
@app.route("/")
def index():
    watchlist = []
    if current_user.is_authenticated:
        watchlist = [item.ticker for item in current_user.watchlist]
    return render_template("markets.html", watchlist=watchlist)


@app.route("/about")
def about():
    return render_template("landing.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name             = request.form.get("name", "").strip()
        email            = request.form.get("email", "").strip().lower()
        password         = request.form.get("password", "")
        experience_level = request.form.get("experience_level", "beginner")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html")

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("signup.html")

        user = User(name=name, email=email, experience_level=experience_level)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        # Add default watchlist for new users
        defaults = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
        for ticker in defaults:
            db.session.add(WatchlistItem(user_id=user.id, ticker=ticker))
        db.session.commit()

        login_user(user)
        flash("Welcome to Merdian Markets!", "success")
        return redirect(url_for("dashboard"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user     = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


# ============================================================
# ROUTES - Dashboard
# ============================================================
@app.route("/dashboard")
@login_required
def dashboard():
    watchlist = [item.ticker for item in current_user.watchlist]
    return render_template("dashboard.html", user=current_user, watchlist=watchlist)


@app.route("/watchlist/add", methods=["POST"])
@login_required
def add_ticker():
    ticker = request.form.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    existing = WatchlistItem.query.filter_by(user_id=current_user.id, ticker=ticker).first()
    if existing:
        return jsonify({"error": "Ticker already in watchlist"}), 400

    if len(current_user.watchlist) >= 50:
        return jsonify({"error": "Watchlist limit is 50 tickers"}), 400

    db.session.add(WatchlistItem(user_id=current_user.id, ticker=ticker))
    db.session.commit()
    return jsonify({"success": True, "ticker": ticker})


@app.route("/watchlist/remove", methods=["POST"])
@login_required
def remove_ticker():
    ticker = request.form.get("ticker", "").strip().upper()
    item   = WatchlistItem.query.filter_by(user_id=current_user.id, ticker=ticker).first()
    if item:
        db.session.delete(item)
        db.session.commit()
    return jsonify({"success": True})


# ============================================================
# ROUTES - Data API (called by frontend via JavaScript)
# ============================================================
@app.route("/api/stock/<ticker>")
@login_required
def api_stock(ticker):
    from data import get_price_data, get_fundamentals, get_earnings_date, get_news
    ticker = ticker.upper()
    return jsonify({
        "price":        get_price_data(ticker),
        "fundamentals": get_fundamentals(ticker),
        "earnings":     get_earnings_date(ticker),
        "news":         get_news(ticker)
    })


@app.route("/api/ai/<ticker>")
@login_required
def api_ai_insights(ticker):
    if not current_user.is_pro:
        return jsonify({"error": "pro_required"}), 403

    from data import get_price_data, get_fundamentals, get_earnings_date, get_news
    from ai   import get_stock_deep_dive
    from concurrent.futures import ThreadPoolExecutor

    ticker = ticker.upper()
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_price    = ex.submit(get_price_data,    ticker)
        f_fund     = ex.submit(get_fundamentals,  ticker)
        f_earnings = ex.submit(get_earnings_date, ticker)
        f_news     = ex.submit(get_news,          ticker)
        price_data     = f_price.result()
        fundamentals   = f_fund.result()
        earnings_date  = f_earnings.result()
        news_headlines = f_news.result()

    from data import _get, _set
    cache_key = f"ai:stock:{ticker}"
    cached = _get(cache_key)
    if cached:
        return jsonify(cached)

    sections, input_tokens, output_tokens = get_stock_deep_dive(
        ticker, price_data, fundamentals, news_headlines, earnings_date
    )
    result = {"sections": sections, "tokens": input_tokens + output_tokens}
    _set(cache_key, result, ttl_seconds=3600)
    return jsonify(result)

@app.route("/api/macro")
@login_required
def api_macro():
    from data import get_macro_summary
    return jsonify(get_macro_summary())


@app.route("/api/heatmap")
@login_required
def api_heatmap():
    from data import get_heatmap_data
    watchlist = [item.ticker for item in current_user.watchlist]
    return jsonify(get_heatmap_data(watchlist))


@app.route("/api/stock/<ticker>/extended")
@login_required
def api_stock_extended(ticker):
    from data import get_price_data, get_fundamentals, get_earnings_date, get_news
    from ai import METRIC_EXPLAINERS
    ticker = ticker.upper()
    fundamentals = get_fundamentals(ticker)
    return jsonify({
        "price":        get_price_data(ticker),
        "fundamentals": fundamentals,
        "earnings":     get_earnings_date(ticker),
        "news":         get_news(ticker),
        "explainers":   METRIC_EXPLAINERS
    })


@app.route("/api/industry/<sector>")
@login_required
def api_industry(sector):
    from data import get_industry_news
    return jsonify(get_industry_news(sector))


# ============================================================
# ROUTES - Payments (Stripe)
# ============================================================
@app.route("/donate")
@login_required
def donate():
    return render_template("donate.html", user=current_user)


@app.route("/donate/checkout", methods=["POST"])
@login_required
def donate_checkout():
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    amount         = int(request.form.get("amount", 5)) * 100  # convert to cents

    try:
        checkout = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[{
                "price_data": {
                    "currency":     "usd",
                    "unit_amount":  amount,
                    "product_data": {
                        "name": "Merdian Markets - Support",
                        "description": "Thank you for supporting Merdian Markets! Your donation unlocks AI features."
                    }
                },
                "quantity": 1
            }],
            mode="payment",
            success_url=url_for("donate_success", _external=True) + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=url_for("donate", _external=True),
            metadata={"user_id": current_user.id}
        )
        return redirect(checkout.url)
    except Exception as e:
        flash("Payment error: " + str(e), "error")
        return redirect(url_for("donate"))


@app.route("/donate/success")
@login_required
def donate_success():
    import stripe
    stripe.api_key  = os.getenv("STRIPE_SECRET_KEY")
    session_id      = request.args.get("session_id")

    try:
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        if checkout_session.payment_status == "paid":
            current_user.is_pro = True
            log = DonationLog(
                user_id=current_user.id,
                amount_cents=checkout_session.amount_total,
                stripe_session=session_id
            )
            db.session.add(log)
            db.session.commit()
            flash("Thank you! AI features are now unlocked.", "success")
    except Exception as e:
        flash("Could not verify payment: " + str(e), "error")

    return redirect(url_for("dashboard"))


# ============================================================
# ROUTES - Settings
# ============================================================
@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if request.method == "POST":
        current_user.name             = request.form.get("name", current_user.name).strip()
        current_user.experience_level = request.form.get("experience_level", current_user.experience_level)
        db.session.commit()
        flash("Settings saved.", "success")
    return render_template("settings.html", user=current_user)

# ============================================================
# SIMULATOR ROUTES
# ============================================================
@app.route("/simulator")
@login_required
def simulator():
    from trading_bot import get_portfolio_stats, get_sp500_return
    portfolio_row = SimulatorPortfolio.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(SimulatorPortfolio.created_at.desc()).first()
    portfolio = json.loads(portfolio_row.portfolio_json) if portfolio_row else None
    stats     = get_portfolio_stats(portfolio) if portfolio else None
    return render_template("simulator.html",
        portfolio    = portfolio,
        stats        = stats,
        has_portfolio= portfolio is not None
    )


@app.route("/simulator/start", methods=["POST"])
@login_required
def simulator_start():
    from trading_bot import create_new_portfolio
    existing = SimulatorPortfolio.query.filter_by(user_id=current_user.id, is_active=True).all()
    for p in existing:
        p.is_active = False
    portfolio     = create_new_portfolio(current_user.id)
    portfolio_row = SimulatorPortfolio(
        user_id        = current_user.id,
        portfolio_json = json.dumps(portfolio),
        is_active      = True
    )
    db.session.add(portfolio_row)
    db.session.commit()
    return jsonify({"success": True})


@app.route("/simulator/run", methods=["POST"])
@login_required
def simulator_run():
    from trading_bot import run_bot_cycle, get_portfolio_stats
    portfolio_row = SimulatorPortfolio.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(SimulatorPortfolio.created_at.desc()).first()
    if not portfolio_row:
        return jsonify({"error": "No active portfolio"}), 400
    portfolio          = json.loads(portfolio_row.portfolio_json)
    portfolio, actions = run_bot_cycle(portfolio)
    stats              = get_portfolio_stats(portfolio)
    portfolio_row.portfolio_json = json.dumps(portfolio)
    portfolio_row.updated_at     = datetime.utcnow()
    db.session.commit()
    return jsonify({"success": True, "actions": actions, "stats": stats,
                    "positions": portfolio["positions"], "cash": round(portfolio["cash"], 2)})


@app.route("/api/simulator/status")
@login_required
def api_simulator_status():
    from trading_bot import get_portfolio_stats
    portfolio_row = SimulatorPortfolio.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(SimulatorPortfolio.created_at.desc()).first()
    if not portfolio_row:
        return jsonify({"has_portfolio": False})
    portfolio = json.loads(portfolio_row.portfolio_json)
    stats     = get_portfolio_stats(portfolio)
    return jsonify({"has_portfolio": True, "stats": stats,
                    "positions": portfolio["positions"],
                    "trade_history": portfolio["trade_history"][-10:],
                    "last_updated": portfolio["last_updated"]})


@app.route("/api/simulator/history")
@login_required
def api_simulator_history():
    portfolio_row = SimulatorPortfolio.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(SimulatorPortfolio.created_at.desc()).first()
    if not portfolio_row:
        return jsonify({"trades": []})
    portfolio = json.loads(portfolio_row.portfolio_json)
    return jsonify({"trades": portfolio["trade_history"]})


@app.route("/simulator/reset", methods=["POST"])
@login_required
def simulator_reset():
    existing = SimulatorPortfolio.query.filter_by(user_id=current_user.id, is_active=True).all()
    for p in existing:
        p.is_active = False
    db.session.commit()
    return jsonify({"success": True})

@app.route("/simulator/performance")
@login_required
def simulator_performance():
    return render_template("performance.html")


@app.route("/api/simulator/performance")
@login_required
def api_simulator_performance():
    from trading_bot import get_portfolio_stats, get_sp500_return
    portfolio_row = SimulatorPortfolio.query.filter_by(
        user_id=current_user.id, is_active=True
    ).order_by(SimulatorPortfolio.created_at.desc()).first()
    if not portfolio_row:
        return jsonify({"error": "No portfolio found"}), 404
    portfolio    = json.loads(portfolio_row.portfolio_json)
    stats        = get_portfolio_stats(portfolio)
    trade_history = portfolio.get("trade_history", [])

    # Sector P&L breakdown
    sector_pnl = {}
    for trade in trade_history:
        sector = trade.get("sector", "Unknown")
        sector_pnl[sector] = sector_pnl.get(sector, 0) + trade.get("pnl", 0)

    # Best and worst trades
    closed = [t for t in trade_history if t.get("pnl") is not None]
    best  = max(closed, key=lambda t: t.get("pnl", 0), default=None)
    worst = min(closed, key=lambda t: t.get("pnl", 0), default=None)

    # Avg hold time
    hold_days = []
    for t in closed:
        try:
            entry = datetime.fromisoformat(t["entry_date"])
            exit_ = datetime.fromisoformat(t["exit_date"])
            hold_days.append((exit_ - entry).days)
        except:
            pass
    avg_hold = round(sum(hold_days) / len(hold_days), 1) if hold_days else 0

    import json as _json
    def safe(obj):
        """Recursively convert non-serializable types to JSON-safe equivalents."""
        if isinstance(obj, dict):
            return {k: safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe(i) for i in obj]
        elif isinstance(obj, bool):
            return bool(obj)
        elif hasattr(obj, 'item'):   # numpy types
            return obj.item()
        return obj

    return jsonify(safe({
        "stats":         stats,
        "equity_curve":  portfolio.get("equity_curve", []),
        "trade_history": trade_history,
        "sector_pnl":    [{"sector": k, "pnl": round(v, 2)} for k, v in sector_pnl.items()],
        "best_trade":    best,
        "worst_trade":   worst,
        "avg_hold_days": avg_hold,
        "sp500_return":  get_sp500_return(portfolio.get("created_at", "")),
    }))

# ============================================================




# ============================================================
# STOCK DETAIL PAGE
# ============================================================
@app.route("/stock/<ticker>")
def stock_detail(ticker):
    ticker = ticker.upper()
    return render_template("stock.html", ticker=ticker,
                           is_pro=current_user.is_pro if current_user.is_authenticated else False,
                           is_authenticated=current_user.is_authenticated,
                           experience_level=current_user.experience_level if current_user.is_authenticated else "beginner",
                           user_name=current_user.name if current_user.is_authenticated else None)


@app.route("/api/sector/<sector_name>")
def api_sector_stocks(sector_name):
    """Return top stocks for a given sector with price data."""
    from data import get_sector_stocks
    return jsonify(get_sector_stocks(sector_name))


@app.route("/api/chart/<ticker>")
def api_chart(ticker):
    """Return OHLC chart data for a given period/interval."""
    import yfinance as yf
    ticker   = ticker.upper()
    period   = request.args.get("period",   "1d")
    interval = request.args.get("interval", "5m")
    try:
        stock      = yf.Ticker(ticker)
        hist       = stock.history(period=period, interval=interval)
        if len(hist) == 0:
            return jsonify({"error": "No data"}), 404
        prices     = [round(float(p), 2) for p in hist["Close"].tolist()]
        timestamps = [str(t) for t in hist.index.tolist()]
        return jsonify({"prices": prices, "timestamps": timestamps})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist/check/<ticker>")
@login_required
def api_watchlist_check(ticker):
    ticker      = ticker.upper()
    in_watchlist = any(item.ticker == ticker for item in current_user.watchlist)
    return jsonify({"in_watchlist": in_watchlist})


# ============================================================
# AI ANALYSIS ROUTES
# ============================================================
@app.route("/api/ai/sector/<sector_name>")
def api_ai_sector(sector_name):
    if not current_user.is_authenticated or not current_user.is_pro:
        return jsonify({"error": "pro_required"}), 403
    try:
        from data import get_sector_stocks, SECTOR_ETFS, get_sector_heatmap
        from ai import get_sector_analysis
        stocks       = get_sector_stocks(sector_name)
        heatmap      = get_sector_heatmap()
        etf_data     = next((s for s in heatmap if s["label"] == sector_name), None)
        etf_change   = etf_data["change_pct"] if etf_data else 0
        from data import _get, _set
        cache_key = f"ai:sector:{sector_name}"
        cached = _get(cache_key)
        if cached:
            return jsonify(cached)
        sections = get_sector_analysis(sector_name, stocks, etf_change)
        result   = {"sections": sections}
        _set(cache_key, result, ttl_seconds=1800)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/ai/macro")
def api_ai_macro():
    if not current_user.is_authenticated or not current_user.is_pro:
        return jsonify({"error": "pro_required"}), 403
    try:
        from data import get_macro_summary
        from ai import get_macro_analysis
        import yfinance as yf
        macro_data = get_macro_summary()
        vix        = yf.Ticker("^VIX").history(period="1d")
        vix_level  = round(float(vix["Close"].iloc[-1]), 2) if len(vix) > 0 else 20
        from data import _get, _set
        cache_key = "ai:macro"
        cached = _get(cache_key)
        if cached:
            return jsonify(cached)
        sections = get_macro_analysis(macro_data, vix_level)
        result   = {"sections": sections, "vix": vix_level}
        _set(cache_key, result, ttl_seconds=900)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/portfolio")
@login_required
def api_ai_portfolio():
    if not current_user.is_pro:
        return jsonify({"error": "pro_required"}), 403
    try:
        from data import get_price_data, get_fundamentals
        from ai import get_portfolio_analysis
        watchlist    = [item.ticker for item in current_user.watchlist]
        if not watchlist:
            return jsonify({"error": "empty_watchlist"}), 400
        from concurrent.futures import ThreadPoolExecutor

        def fetch_ticker_data(t):
            price = get_price_data(t)
            fund  = get_fundamentals(t)
            return t, price, fund

        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(fetch_ticker_data, watchlist))

        watchlist_data = []
        for ticker, price, fund in results:
            if price and "error" not in price:
                watchlist_data.append({
                    "ticker":     ticker,
                    "price":      price.get("price"),
                    "change_pct": price.get("change_pct"),
                    "ma_context": price.get("ma_context"),
                    "vol_signal": price.get("vol_signal"),
                    "sector":     fund.get("sector", "Unknown") if fund and "error" not in fund else "Unknown",
                    "signal":     fund.get("signal", "N/A")     if fund and "error" not in fund else "N/A",
                })
        from data import _get, _set
        cache_key = f"ai:portfolio:{current_user.id}"
        cached = _get(cache_key)
        if cached:
            return jsonify(cached)
        sections = get_portfolio_analysis(watchlist_data)
        result   = {"sections": sections, "stocks_analyzed": len(watchlist_data)}
        _set(cache_key, result, ttl_seconds=1800)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================
# ROUTES - Markets Home Page
# ============================================================
@app.route("/markets")
def markets():
    watchlist = []
    is_pro    = False
    if current_user.is_authenticated:
        watchlist = [item.ticker for item in current_user.watchlist]
        is_pro    = current_user.is_pro
    return render_template("markets.html", watchlist=watchlist, is_pro=is_pro)


@app.route("/api/markets/indices")
def api_indices():
    try:
        from data import get_indices_chart_data
        return jsonify(get_indices_chart_data())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/markets/news")
def api_markets_news():
    try:
        from data import get_market_news
        news = get_market_news()
        return jsonify(news)
    except Exception as e:
        return jsonify([]), 200


@app.route("/api/markets/sectorheatmap")
def api_sector_heatmap():
    try:
        from data import get_sector_heatmap
        return jsonify(get_sector_heatmap())
    except Exception as e:
        return jsonify([]), 200


# ============================================================
# ROUTES - Alpaca Live Paper Trading
# ============================================================

@app.route("/alpaca")
@login_required
def alpaca_dashboard():
    return render_template("alpaca.html")


@app.route("/api/alpaca/status")
@login_required
def api_alpaca_status():
    """Check if Alpaca is connected and return account summary."""
    try:
        from alpaca_bot import check_alpaca_connected, get_alpaca_portfolio
        connected, message = check_alpaca_connected()
        if not connected:
            return jsonify({"connected": False, "error": message})
        portfolio = get_alpaca_portfolio()
        return jsonify({"connected": True, "message": message, "portfolio": portfolio})
    except Exception as e:
        return jsonify({"connected": False, "error": str(e)})


@app.route("/api/alpaca/stats")
@login_required
def api_alpaca_stats():
    """Return full performance stats from Alpaca account."""
    try:
        from alpaca_bot import get_alpaca_stats
        return jsonify(get_alpaca_stats())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpaca/run", methods=["POST"])
@login_required
def api_alpaca_run():
    """Manually trigger one bot cycle against Alpaca."""
    try:
        from alpaca_bot import run_alpaca_cycle
        actions = run_alpaca_cycle()
        return jsonify({
            "success": True,
            "actions": actions,
            "buys":    sum(1 for a in actions if a["type"] == "BUY"),
            "sells":   sum(1 for a in actions if a["type"] == "SELL"),
            "message": f"Cycle complete — {len(actions)} action(s) taken"
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/alpaca/positions")
@login_required
def api_alpaca_positions():
    """Return current open positions from Alpaca."""
    try:
        from alpaca_bot import get_trading_client
        client    = get_trading_client()
        positions = []
        for pos in client.get_all_positions():
            positions.append({
                "ticker":            pos.symbol,
                "qty":               float(pos.qty),
                "entry_price":       round(float(pos.avg_entry_price), 2),
                "current_price":     round(float(pos.current_price), 2),
                "market_value":      round(float(pos.market_value), 2),
                "unrealized_pnl":    round(float(pos.unrealized_pl), 2),
                "unrealized_pnl_pct": round(float(pos.unrealized_plpc) * 100, 2),
                "side":              str(pos.side),
            })
        return jsonify({"positions": positions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpaca/orders")
@login_required
def api_alpaca_orders():
    """Return recent orders from Alpaca."""
    try:
        from alpaca_bot import get_trading_client
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = get_trading_client()
        raw    = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, limit=30))
        orders = []
        for o in raw:
            orders.append({
                "id":          str(o.id),
                "ticker":      o.symbol,
                "side":        str(o.side),
                "qty":         str(o.qty or o.notional),
                "type":        str(o.type),
                "status":      str(o.status),
                "filled_price": round(float(o.filled_avg_price), 2) if o.filled_avg_price else None,
                "submitted_at": str(o.submitted_at),
                "filled_at":   str(o.filled_at) if o.filled_at else None,
            })
        return jsonify({"orders": orders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/alpaca/close/<ticker>", methods=["POST"])
@login_required
def api_alpaca_close(ticker):
    """Manually close a specific position."""
    try:
        from alpaca_bot import submit_sell_order
        result = submit_sell_order(ticker, "Manually closed via dashboard")
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    
# ============================================================
# LIVE PRICES — Server-Sent Events
# ============================================================
@app.route("/api/live-prices")
@login_required
def api_live_prices():
    """SSE stream — browser connects once, gets price updates pushed every 5s."""
    from data import get_live_prices_batch, HEATMAP_EXTRAS
    import json as _json

    watchlist = [item.ticker for item in current_user.watchlist]
    tickers   = list(dict.fromkeys(watchlist + HEATMAP_EXTRAS))

    def stream():
        import time
        while True:
            prices = get_live_prices_batch(tickers)
            yield f"data: {_json.dumps(prices)}\n\n"
            time.sleep(5)

    from flask import Response
    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/live-prices/snapshot")
@login_required
def api_live_prices_snapshot():
    """One-shot snapshot of current live prices."""
    from data import get_live_prices_batch, HEATMAP_EXTRAS
    watchlist = [item.ticker for item in current_user.watchlist]
    tickers   = list(dict.fromkeys(watchlist + HEATMAP_EXTRAS))
    return jsonify(get_live_prices_batch(tickers))
    

# ============================================================
# INIT
# ============================================================


if __name__ == "__main__":
    with app.app_context():
        db.create_all()

        # Start Alpaca WebSocket stream for live prices
        try:
            from data import start_price_stream, HEATMAP_EXTRAS
            from trading_bot import TRADEABLE_UNIVERSE
            stream_tickers = list(set(HEATMAP_EXTRAS + TRADEABLE_UNIVERSE))
            start_price_stream(stream_tickers)
        except Exception as e:
            print(f"WebSocket stream not started: {e}")

        # Start bot scheduler
        try:
            from trading_bot import init_scheduler
            init_scheduler(app, db, SimulatorPortfolio)
        except Exception as e:
            print(f"Scheduler not started: {e}")

    app.run(debug=True, port=8000, threaded=True)