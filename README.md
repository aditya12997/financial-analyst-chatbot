# Financial Analyst Chatbot

A multi-tool LangGraph chatbot that answers questions about stocks, market
sentiment, and financial news using live data, not guesses. Built on Groq,
with tools for stock quotes, price charts, company fundamentals, portfolio
performance, currency conversion, and sentiment analysis on top of general
search, Wikipedia, and arXiv.

## Notebooks

- **`financial_market_assistant.ipynb`** — the first version. Introduces the
  core loop-based architecture (the classroom tool-calling tutorial dead-ends
  after one tool call; this version loops results back to the model so it
  can actually synthesize an answer), plus live stock quotes, a market
  overview, price charts, and currency conversion.

- **`financial_analyst_chatbot.ipynb`** — the extended version. Adds company
  fundamentals, a portfolio performance tool, multi-turn conversation memory
  via a checkpointer, a market sentiment tool (with its own internal LLM
  call, separate from the main model), a disclaimer guardrail enforced as a
  graph node rather than just a prompt instruction, and a companion
  Streamlit app generated directly from the notebook.

## Stack

LangGraph, LangChain, Groq (`openai/gpt-oss-120b`), Tavily, yfinance,
matplotlib, Pydantic for structured output.

## Run it

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add free API keys:
   [Groq](https://console.groq.com/keys), [Tavily](https://tavily.com)
3. Open either notebook and run all cells

## Note on rate limits

Groq's free tier caps token usage per minute. Running many examples back to
back, especially the sentiment tool (which makes two model calls per
question), can hit that limit. Re-running the cell after a short pause
resolves it.