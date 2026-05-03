"""
Finance Research Pipeline
=========================
Production-style market sentiment analysis pipeline.
Stack: LangChain + Google Gemini 2.5 Flash + yfinance + DuckDuckGo + MLflow

Steps:
  1. accept_input      – validate company name
  2. resolve_ticker    – map company → stock symbol
  3. fetch_news        – DuckDuckGo search for recent headlines
  4. fetch_stock_price – yfinance for price & history
  5. analyze_sentiment – Gemini 2.5 Flash for NLP + JSON output
  6. generate_output   – assemble final research report

Usage:
  python pipeline.py --company "Tesla"
  python pipeline.py --company "Reliance Industries"
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime

import requests
import yfinance as yf
import mlflow
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.tools import DuckDuckGoSearchResults
from pydantic import BaseModel, Field

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ── env ───────────────────────────────────────────────────────────────────────
GOOGLE_API_KEY     = os.getenv("GOOGLE_API_KEY", "")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "mlruns")   # local by default

# ── static ticker fallback ────────────────────────────────────────────────────
TICKER_MAP = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "meta": "META",
    "facebook": "META", "tesla": "TSLA", "nvidia": "NVDA",
    "netflix": "NFLX", "uber": "UBER", "airbnb": "ABNB",
    "reliance": "RELIANCE.NS", "reliance industries": "RELIANCE.NS",
    "tata consultancy": "TCS.NS", "tcs": "TCS.NS",
    "infosys": "INFY", "wipro": "WIT",
    "jpmorgan": "JPM", "jp morgan": "JPM",
    "goldman sachs": "GS", "berkshire": "BRK-B",
}

# ── Pydantic schema for structured output ─────────────────────────────────────
class ResearchOutput(BaseModel):
    company_name: str               = Field(description="Full company name")
    stock_code: str                 = Field(description="Stock ticker symbol")
    current_price: float            = Field(description="Current stock price")
    price_change_7d: str            = Field(description="7-day % change e.g. +2.4%")
    price_change_30d: str           = Field(description="30-day % change e.g. -1.1%")
    fifty_two_week_high: float      = Field(description="52-week high")
    fifty_two_week_low: float       = Field(description="52-week low")
    news_summary: str               = Field(description="2-3 sentence news summary")
    sentiment: str                  = Field(description="Positive / Negative / Neutral")
    confidence_score: float         = Field(description="Confidence 0.0–1.0")
    people_names: list[str]         = Field(description="Named people in news")
    places_names: list[str]         = Field(description="Places mentioned")
    other_companies_referred: list[str] = Field(description="Other companies mentioned")
    related_industries: list[str]   = Field(description="Related industries")
    market_implications: str        = Field(description="Market implications")
    investment_recommendation: str  = Field(description="Buy / Hold / Avoid")
    recommendation_rationale: str   = Field(description="Rationale for recommendation")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 – accept_input
# ═══════════════════════════════════════════════════════════════════════════════
def accept_input(company_name: str) -> dict:
    log.info(f"[Step 1] Input: {company_name!r}")
    if not company_name or not company_name.strip():
        raise ValueError("Company name must not be empty.")
    return {"company_name": company_name.strip()}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 – resolve_ticker
# ═══════════════════════════════════════════════════════════════════════════════
def resolve_ticker(state: dict, llm: ChatGoogleGenerativeAI) -> dict:
    company = state["company_name"]
    log.info(f"[Step 2] Resolving ticker for: {company!r}")

    # 1) static dict
    ticker = TICKER_MAP.get(company.lower())
    if ticker:
        log.info(f"  Static dict → {ticker}")
        state.update(ticker=ticker, ticker_source="static_dict")
        return state

    # 2) LLM
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Return ONLY the primary stock ticker symbol. No explanation."),
            ("human", "Stock ticker for: {company}")
        ])
        result = (prompt | llm).invoke({"company": company})
        candidate = result.content.strip().upper().split()[0]
        # validate via yfinance fast_info
        fi = yf.Ticker(candidate).fast_info
        if fi.get("lastPrice") or fi.get("regularMarketPrice"):
            log.info(f"  LLM → {candidate}")
            state.update(ticker=candidate, ticker_source="llm")
            return state
    except Exception as e:
        log.warning(f"  LLM resolution failed: {e}")

    # 3) Yahoo Finance search API
    try:
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search"
            f"?q={requests.utils.quote(company)}&quotesCount=1&lang=en-US"
        )
        quotes = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8).json().get("quotes", [])
        if quotes:
            ticker = quotes[0]["symbol"]
            log.info(f"  Yahoo search → {ticker}")
            state.update(ticker=ticker, ticker_source="yahoo_search")
            return state
    except Exception as e:
        log.warning(f"  Yahoo search failed: {e}")

    raise ValueError(f"Could not resolve ticker for '{company}'.")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 – fetch_news
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_news(state: dict) -> dict:
    company = state["company_name"]
    log.info(f"[Step 3] Fetching news for: {company!r}")

    try:
        tool = DuckDuckGoSearchResults(num_results=8, output_format="list")
        results = tool.invoke(f"{company} stock news 2026")
        items = results if isinstance(results, list) else []
        log.info(f"  Fetched {len(items)} articles")
    except Exception as e:
        log.warning(f"  DuckDuckGo failed: {e}")
        items = []

    state["news_items"] = items[:8]
    state["news_raw"] = "\n".join(
        f"- {r.get('title','')}: {r.get('snippet','')}" for r in items
    ) or f"No recent news found for {company}."
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 – fetch_stock_price
# ═══════════════════════════════════════════════════════════════════════════════
def fetch_stock_price(state: dict) -> dict:
    sym = state["ticker"]
    log.info(f"[Step 4] Fetching stock data: {sym}")

    try:
        tk = yf.Ticker(sym)
        info = tk.info

        price = (info.get("currentPrice")
                 or info.get("regularMarketPrice")
                 or info.get("previousClose", 0.0))

        def pct(period):
            h = tk.history(period=period)
            if h.empty or len(h) < 2:
                return "N/A"
            chg = (h["Close"].iloc[-1] - h["Close"].iloc[0]) / h["Close"].iloc[0] * 100
            return f"{chg:+.1f}%"

        state["stock_data"] = {
            "current_price":      round(float(price), 2),
            "price_change_7d":    pct("7d"),
            "price_change_30d":   pct("30d"),
            "fifty_two_week_high": round(float(info.get("fiftyTwoWeekHigh", 0)), 2),
            "fifty_two_week_low":  round(float(info.get("fiftyTwoWeekLow", 0)), 2),
            "long_name":          info.get("longName", state["company_name"]),
        }
        log.info(f"  Price: ${price:.2f}")
    except Exception as e:
        log.warning(f"  yfinance error: {e}")
        state["stock_data"] = {
            "current_price": 0.0, "price_change_7d": "N/A", "price_change_30d": "N/A",
            "fifty_two_week_high": 0.0, "fifty_two_week_low": 0.0,
            "long_name": state["company_name"],
        }
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 – analyze_sentiment
# ═══════════════════════════════════════════════════════════════════════════════
ANALYSIS_PROMPT = """\
You are a senior financial analyst. Analyse the following news and stock data for {company}.

## Recent News Headlines
{news}

## Stock Data
- Ticker: {ticker}
- Current Price: ${current_price}
- 7-day change: {price_7d}
- 30-day change: {price_30d}
- 52-week high: ${high} | low: ${low}

Respond ONLY with a valid JSON object matching this exact schema (no markdown, no extra keys):
{{
  "company_name": str,
  "stock_code": str,
  "current_price": float,
  "price_change_7d": str,
  "price_change_30d": str,
  "fifty_two_week_high": float,
  "fifty_two_week_low": float,
  "news_summary": str,
  "sentiment": "Positive" | "Negative" | "Neutral",
  "confidence_score": float (0.0–1.0),
  "people_names": [str],
  "places_names": [str],
  "other_companies_referred": [str],
  "related_industries": [str],
  "market_implications": str,
  "investment_recommendation": "Buy" | "Hold" | "Avoid",
  "recommendation_rationale": str
}}
"""

def analyze_sentiment(state: dict, llm: ChatGoogleGenerativeAI) -> dict:
    log.info("[Step 5] Running sentiment analysis via Gemini 2.5 Flash")
    sd = state["stock_data"]

    prompt = ANALYSIS_PROMPT.format(
        company=state["company_name"],
        news=state["news_raw"],
        ticker=state["ticker"],
        current_price=sd["current_price"],
        price_7d=sd["price_change_7d"],
        price_30d=sd["price_change_30d"],
        high=sd["fifty_two_week_high"],
        low=sd["fifty_two_week_low"],
    )

    parser = JsonOutputParser(pydantic_object=ResearchOutput)
    chain  = ChatPromptTemplate.from_messages([("human", "{prompt}")]) | llm | parser

    result = chain.invoke({"prompt": prompt})
    log.info(f"  Sentiment: {result.get('sentiment')} | Recommendation: {result.get('investment_recommendation')}")
    state["analysis"] = result
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 – generate_output
# ═══════════════════════════════════════════════════════════════════════════════
def generate_output(state: dict) -> dict:
    log.info("[Step 6] Generating final research report")
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "ticker_source": state.get("ticker_source"),
        **state["analysis"],
    }
    state["report"] = report
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def run_pipeline(company_name: str) -> dict:
    if not GOOGLE_API_KEY:
        raise EnvironmentError("Set GOOGLE_API_KEY environment variable.")

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.1,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("finance_research_pipeline")

    with mlflow.start_run(run_name=f"{company_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_param("company_name", company_name)
        mlflow.log_param("model", "gemini-2.5-flash")
        mlflow.log_param("news_tool", "DuckDuckGo")

        state = {}
        steps = [
            ("accept_input",      lambda s: accept_input(company_name)),
            ("resolve_ticker",    lambda s: resolve_ticker(s, llm)),
            ("fetch_news",        fetch_news),
            ("fetch_stock_price", fetch_stock_price),
            ("analyze_sentiment", lambda s: analyze_sentiment(s, llm)),
            ("generate_output",   generate_output),
        ]

        for step_name, fn in steps:
            t0 = time.time()
            with mlflow.start_span(name=step_name):          # tracing span per step
                if step_name == "accept_input":
                    state = fn(state)
                else:
                    state = fn(state)
            elapsed = time.time() - t0
            mlflow.log_metric(f"{step_name}_seconds", round(elapsed, 3))
            log.info(f"  ✓ {step_name} ({elapsed:.2f}s)")

        report = state["report"]

        # log key metrics
        mlflow.log_metric("confidence_score", report.get("confidence_score", 0))
        mlflow.log_param("sentiment", report.get("sentiment"))
        mlflow.log_param("recommendation", report.get("investment_recommendation"))

        # save JSON artifact
        out_path = f"output_{company_name.replace(' ', '_')}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(out_path)
        log.info(f"Report saved → {out_path}")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finance Research Pipeline")
    parser.add_argument("--company", required=True, help='Company name e.g. "Tesla"')
    args = parser.parse_args()

    report = run_pipeline(args.company)
    print("\n" + "═" * 60)
    print("RESEARCH REPORT")
    print("═" * 60)
    print(json.dumps(report, indent=2))
