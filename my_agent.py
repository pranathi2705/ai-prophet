import os
import re
import json
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENAI_API_KEY")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")


def web_search(query: str) -> str:
    """Search the web using Brave Search API."""
    if not BRAVE_API_KEY:
        return "Web search unavailable."
    try:
        response = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": BRAVE_API_KEY
            },
            params={"q": query, "count": 5},
            timeout=5
        )
        results = response.json().get("web", {}).get("results", [])
        snippets = [f"- {r['title']}: {r['description']}" for r in results[:5]]
        return "\n".join(snippets) if snippets else "No results found."
    except Exception:
        return "Web search failed."


def get_kalshi_price(ticker: str) -> dict | None:
    """Fetch current market prices from Kalshi."""
    try:
        url = f"https://api.kalshi.com/trade-api/v2/markets/{ticker}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            market = data.get("market", {})
            yes_bid = market.get("yes_bid", None)
            yes_ask = market.get("yes_ask", None)
            if yes_bid and yes_ask:
                yes_price = (yes_bid + yes_ask) / 2 / 100
                return {"yes": round(yes_price, 3), "no": round(1 - yes_price, 3)}
    except Exception:
        pass
    return None


def clean_predictions(result: dict, valid_outcomes: list) -> dict:
    """Remove invalid outcomes and redistribute their probability."""
    probs = result.get("probabilities", [])

    # Separate valid and invalid
    valid = [p for p in probs if p["market"] in valid_outcomes]
    invalid_total = sum(p["probability"] for p in probs if p["market"] not in valid_outcomes)

    if not valid:
        n = len(valid_outcomes)
        return {"probabilities": [{"market": o, "probability": round(1/n, 4)} for o in valid_outcomes]}

    # Redistribute invalid probability equally to valid ones
    bonus = invalid_total / len(valid)
    for p in valid:
        p["probability"] = round(p["probability"] + bonus, 4)
    # Normalize so everything sums to exactly 1.0
    total = sum(p["probability"] for p in valid)
    for p in valid:
        p["probability"] = max(round(p["probability"] / total, 4), 0.001)
    # Make sure all valid outcomes are included
    covered = {p["market"] for p in valid}
    for o in valid_outcomes:
        if o not in covered:
            valid.append({"market": o, "probability": 0.001})

    return {"probabilities": valid}


def predict(event: dict) -> dict:
    title = event.get("title", "")
    description = event.get("description", "")
    outcomes = event.get("outcomes", [])
    rules = event.get("rules", "")
    ticker = event.get("market_ticker", "")
    category = event.get("category", "")

    outcomes_text = "\n".join([f"- {o}" for o in outcomes])

    # 1. Get Kalshi price
    kalshi_data = get_kalshi_price(ticker)
    if kalshi_data:
        kalshi_context = f"""
KALSHI MARKET PRICE (real money traders RIGHT NOW):
- YES: {kalshi_data['yes']*100:.1f}%
- NO: {kalshi_data['no']*100:.1f}%
Use this as your primary anchor.
"""
    else:
        kalshi_context = "Kalshi price not available."

    # 2. Search web for current info
    search_query = f"{title} {category} 2026 current"
    search_results = web_search(search_query)

    # For events with many outcomes, do a second focused search
    if len(outcomes) > 10:
        focused_query = f"{title} odds favorites 2026"
        focused_results = web_search(focused_query)
        search_results = search_results + "\n" + focused_results

    # 3. Build prompt
    prompt = f"""You are an expert forecaster who predicts outcomes across ANY domain — sports, politics, economics, entertainment, technology, and more.

Event: {title}
Category: {category}
Description: {description}
Rules: {rules}

{kalshi_context}

FRESH WEB SEARCH RESULTS (today's data):
{search_results}

Possible outcomes (YOU MUST ONLY USE THESE EXACT NAMES, NO OTHERS):
{outcomes_text}

STRICT RULES:
- The above list is the COMPLETE list of valid outcomes
- Do NOT add "other", "other outcome", or any name not in the list above
- Every probability must map to one of the exact names listed above
- If you want to assign low probability to many teams, use 0.001 for each — but still name them exactly

Think step by step:

1. UNDERSTAND: What type of event is this? What domain?

2. USE DATA: What does the Kalshi price tell you? What do the web search results say about current standings, polls, odds, or status?

3. REASON:
   - Sports: consider current standings, form, home advantage, injuries
   - Politics: consider polls, incumbency advantage, historical patterns
   - Economics: consider analyst forecasts, current trends, Fed signals
   - Entertainment: consider nominations, critic scores, frontrunners
   - Unknown: use base rates and available evidence

4. CALIBRATE: Don't be overconfident. Spread probability across realistic options.
   IMPORTANT: For events with many outcomes (10+), only assign meaningful probability to the top 5-6 realistic contenders. Give the rest 0.001 each. Keep your JSON clean and valid.

5. OUTPUT only this JSON. Start with {{ and end with }}. No explanation before or after:
{{
  "probabilities": [
    {{"market": "exact outcome name", "probability": 0.65}},
    {{"market": "other outcome", "probability": 0.35}}
  ]
}}

All outcome names must match exactly. Probabilities must sum to 1.0.
NEVER invent outcome names. Only use names from the exact list provided above.
CRITICAL: Return ONLY the JSON object. No explanation text before or after."""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
    )

    content = response.json()["choices"][0]["message"]["content"]

    # Try multiple parsing strategies
    try:
        return clean_predictions(json.loads(content), outcomes)
    except Exception:
        pass

    try:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return clean_predictions(json.loads(match.group()), outcomes)
    except Exception:
        pass

    try:
        cleaned = content.replace('\n', ' ').replace('    ', ' ')
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            return clean_predictions(json.loads(match.group()), outcomes)
    except Exception:
        pass

    # Fallback: equal probabilities
    n = len(outcomes)
    return {"probabilities": [{"market": o, "probability": round(1/n, 4)} for o in outcomes]}