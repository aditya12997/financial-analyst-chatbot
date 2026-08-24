import streamlit as st
import re
import os

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv
load_dotenv()

try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    if "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except FileNotFoundError:
    pass  # no secrets.toml locally, fall back to .env (already loaded above)

import certifi
os.environ['CURL_CA_BUNDLE'] = certifi.where()
os.environ['SSL_CERT_FILE'] = certifi.where()

import curl_cffi.requests as ccr
if not getattr(ccr.Session.__init__, '_is_patched', False):
    _original_session_init = ccr.Session.__init__
    def _patched_session_init(self, *args, **kwargs):
        kwargs.setdefault('verify', False)
        return _original_session_init(self, *args, **kwargs)
    _patched_session_init._is_patched = True
    ccr.Session.__init__ = _patched_session_init

import yfinance as yf
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated


@tool
def get_stock_quote(ticker: str) -> str:
    'Get the current price and daily change for a stock ticker.'
    try:
        info = yf.Ticker(ticker.upper()).fast_info
        change = info.last_price - info.previous_close
        pct = (change / info.previous_close) * 100
        return f'{ticker.upper()}: ${info.last_price:,.2f} ({pct:+.2f}%)'
    except Exception as e:
        return f'Could not retrieve a quote for {ticker}: {e}'


@tool
def plot_stock_chart(ticker: str, period: str = '3mo') -> str:
    'Generate and save a price chart for a stock ticker over a period.'
    import matplotlib.pyplot as plt
    os.makedirs('charts', exist_ok=True)
    try:
        history = yf.Ticker(ticker.upper()).history(period=period)
        if history.empty:
            return f'No data for {ticker}.'
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(history.index, history['Close'])
        ax.set_title(f'{ticker.upper()} - {period}')
        fig.tight_layout()
        path = f'charts/{ticker.upper()}_{period}.png'
        fig.savefig(path, dpi=130)
        plt.close(fig)
        return f'Chart saved to {path}.'
    except Exception as e:
        return f'Could not generate a chart for {ticker}: {e}'


tools = [get_stock_quote, plot_stock_chart]
llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0.2)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = 'You are a financial assistant. Use tools for real prices and charts. Format answers in markdown.'

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def tool_calling_llm(state):
    return {'messages': [llm_with_tools.invoke([SystemMessage(content=SYSTEM_PROMPT)] + state['messages'])]}

builder = StateGraph(State)
builder.add_node('tool_calling_llm', tool_calling_llm)
builder.add_node('tools', ToolNode(tools))
builder.add_edge(START, 'tool_calling_llm')
builder.add_conditional_edges('tool_calling_llm', tools_condition)
builder.add_edge('tools', 'tool_calling_llm')
financial_bot = builder.compile(checkpointer=InMemorySaver())

st.set_page_config(
    page_title='Financial Analyst Chatbot',
    page_icon='📈',
    layout='wide',
)

# ---------------------------------------------------------------------------
# Sidebar: intro, how to use it, credit
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📈 Financial Analyst Chatbot")
    st.markdown(
        "A chat assistant that answers questions about stocks using **live "
        "data**, not guesses. Ask for a price, a chart, or how a stock has "
        "been trending, and it calls real tools to find out before answering."
    )

    st.markdown("---")
    st.markdown("**What it can do**")
    st.markdown(
        "- Look up a stock's current price and daily change\n"
        "- Generate a price history chart for a chosen period\n"
        "- Hold a short back-and-forth conversation with context"
    )

    st.markdown("**Try asking**")
    st.markdown(
        "- *What's Tesla's stock price right now?*\n"
        "- *Show me a 6 month chart for Nvidia*\n"
        "- *How has Apple been doing this year?*"
    )

    st.markdown("---")
    st.caption(
        "This is a demo built for learning and portfolio purposes. It is "
        "general information, not financial advice. Runs on a free API tier, "
        "so it may briefly rate-limit if used heavily."
    )

    st.markdown("---")
    st.markdown("Built by **Aditya Gavali** (AG)")
    st.markdown(
        "[GitHub](https://github.com/aditya12997) · "
        "[LinkedIn](https://www.linkedin.com/in/aditya12997)"
    )

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title('📈 Financial Analyst Chatbot')
st.caption('Live stock prices and charts, powered by LangGraph and Groq. Ask a question below to get started.')

if 'messages' not in st.session_state:
    st.session_state.messages = []

chat_col, chart_col = st.columns([2, 1])

with chat_col:
    if not st.session_state.messages:
        st.info(
            "👋 Ask me about any public stock, for example: "
            "*\"What's the current price of Microsoft?\"* or "
            "*\"Show me a chart for Amazon over the last year.\"*"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    user_input = st.chat_input('Ask about a stock, e.g. Show me a chart for Tesla')

    if user_input:
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.markdown(user_input)

        config = {'configurable': {'thread_id': 'streamlit-session'}}
        with st.spinner('Looking that up...'):
            try:
                result = financial_bot.invoke(
                    {'messages': [{'role': 'user', 'content': user_input}]}, config=config
                )
                answer = result['messages'][-1].content

                for m in result['messages']:
                    if m.type == 'tool' and m.name == 'plot_stock_chart':
                        found = re.search(r'charts/[\w\.]+\.png', m.content)
                        if found:
                            st.session_state['last_chart'] = found.group()
            except Exception as e:
                answer = (
                    "This demo runs on a free API tier and just hit its rate limit. "
                    "Please wait about 20 seconds and try again."
                )

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
        with st.chat_message('assistant'):
            st.markdown(answer)

with chart_col:
    st.subheader('📊 Chart')
    if st.session_state.get('last_chart'):
        st.image(st.session_state['last_chart'])
    else:
        st.caption('A chart will appear here once you ask for one.')
