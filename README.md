# MarketPulse AI

An AI-powered multi-user market intelligence web app built with Flask and Claude AI.

## Features
- Live stock price dashboard for custom watchlists
- AI summaries, buy/hold/watch signals, and sentiment analysis (Pro)
- Earnings calendar with upcoming report dates
- Macro economic overview (S&P, NASDAQ, Gold, Bonds, VIX)
- Industry news by sector
- Personalized trading tips by experience level (Beginner / Intermediate / Advanced)
- Stripe donation to unlock AI features

## Setup Instructions

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Set up environment variables
Copy `.env.example` to `.env` and fill in:
- `SECRET_KEY` — any long random string
- `ANTHROPIC_API_KEY` — from console.anthropic.com
- `STRIPE_SECRET_KEY` — from dashboard.stripe.com (use test keys first)
- `STRIPE_PUBLISHABLE_KEY` — from dashboard.stripe.com

### 3. Run locally
```
python app.py
```
Visit http://localhost:5000

### 4. Deploy to Render (free hosting)
1. Push this folder to a GitHub repo
2. Go to render.com and create a new Web Service
3. Connect your GitHub repo
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Add all environment variables from .env
7. Deploy — you'll get a live URL to share on LinkedIn!

## File Structure
```
market_app/
  app.py          — Flask app, routes, auth, payment handling
  data.py         — Yahoo Finance data fetching (prices, fundamentals, news, earnings, macro)
  ai.py           — Anthropic Claude AI insights generation
  requirements.txt
  .env.example
  templates/
    landing.html  — Public landing/marketing page
    signup.html   — User registration
    login.html    — User login
    dashboard.html — Main live dashboard
    donate.html   — Stripe donation page
```

## Notes
- Free tier: price data, fundamentals, earnings calendar
- Pro (after donation): AI summaries, signals, sentiment, trading tips, macro AI overview
- Yahoo Finance data loads with a small delay between tickers to avoid rate limits
- All AI calls use claude-sonnet-4-6 for cost efficiency (~$0.05-0.10/day for 33 stocks)
