# Finance Research Pipeline

A production-style market sentiment analysis pipeline built with **LangChain + Google Gemini 2.5 Flash + yfinance + MLflow**.

## Stack

| Component | Tool |
|---|---|
| Framework | LangChain |
| LLM | Google Gemini 2.5 Flash |
| News Search | DuckDuckGo (free, no API key) |
| Stock Data | yfinance (free, no API key) |
| Observability | MLflow (local `mlruns/`) |
| Output Parsing | `JsonOutputParser` + Pydantic |

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

Get a free Gemini API key at: https://aistudio.google.com/app/apikey

## Usage

```bash
python pipeline.py --company "Tesla"
python pipeline.py --company "Google"
python pipeline.py --company "Reliance Industries"
```

Output JSON is printed to stdout and saved as `output_<company>.json`.

## Pipeline Steps

```
accept_input → resolve_ticker → fetch_news → fetch_stock_price → analyze_sentiment → generate_output
```

1. **accept_input** – validates and normalises company name
2. **resolve_ticker** – static dict → Gemini LLM → Yahoo Finance search (fallback chain)
3. **fetch_news** – DuckDuckGo search for recent headlines
4. **fetch_stock_price** – current price, 7d/30d change, 52-week range via yfinance
5. **analyze_sentiment** – Gemini 2.5 Flash performs NLP: sentiment, entities, recommendation
6. **generate_output** – assembles final structured JSON report

## MLflow Observability

Each pipeline run is tracked locally in `mlruns/`. To view the UI:

```bash
mlflow ui
# open http://localhost:5000
```

Tracked per run:
- Parameters: `company_name`, `model`, `news_tool`, `sentiment`, `recommendation`
- Metrics: `confidence_score`, per-step execution time (`*_seconds`)
- Artifacts: output JSON file
- Spans: one MLflow span per pipeline step

## Output Schema

See `sample_output.json` for a full example. Key fields:

```json
{
  "company_name": "Google",
  "stock_code": "GOOGL",
  "current_price": 178.25,
  "price_change_7d": "+2.4%",
  "price_change_30d": "-1.1%",
  "fifty_two_week_high": 191.75,
  "fifty_two_week_low": 130.67,
  "news_summary": "...",
  "sentiment": "Positive",
  "confidence_score": 0.85,
  "people_names": [...],
  "places_names": [...],
  "other_companies_referred": [...],
  "related_industries": [...],
  "market_implications": "...",
  "investment_recommendation": "Buy",
  "recommendation_rationale": "..."
}
```
