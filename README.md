# LLM-Driven Automated Trading Agent

A modular course project that combines **financial news understanding** with **market trend filtering** to produce a paper-trading signal and a basic backtest. The system is intentionally split into small classes and scripts so each responsibility is isolated: data collection, sentiment analysis, technical features, signal generation, execution, and evaluation.

This repository is designed to satisfy the common rubric expectations of an LLM-driven finance assignment:
- a reproducible data pipeline,
- a clear role for the LLM or NLP model,
- explainable trading logic,
- risk controls,
- backtesting and benchmarking,
- and a maintainable repository structure.

---

## 1. Project goal

The project answers a simple but defensible question:

> Can we use an LLM or finance-specific NLP model as the **"brain"** for interpreting recent news, and combine that with a lightweight **technical trend filter** as the **"body"** to generate a more disciplined trading decision?

Instead of asking the model to predict prices directly, this repository uses the model only for a task that is more appropriate for language models:

- read recent stock-related headlines,
- classify sentiment,
- convert that into a signed sentiment score and a conviction score,
- and pass that information into a deterministic trading rule.

The actual trade decision is **not** made by the model alone. A trade is only allowed when:

1. the market trend condition is favorable, and  
2. the news sentiment condition is favorable.

That separation is important because it keeps the pipeline more explainable and more aligned with the assignment prompt.

---

## 2. High-level system design

The repository follows this flow:

```text
Market Data (Yahoo Finance)
        +
Recent News (Google News RSS or Alpaca News)
        |
        v
Sentiment Layer
  - Local FinBERT   OR
  - OpenAI JSON sentiment agent
        |
        v
Signal Layer
  - 50-day SMA trend filter
  - average sentiment filter
  - conviction threshold
        |
        v
Decision
  - BUY / HOLD / SELL
        |
        +--> Paper execution via Alpaca bracket order
        |
        +--> Backtest with benchmark comparison
```

The practical interpretation is:
- **trend filter** says whether the stock is in an upward or downward technical regime,
- **sentiment layer** says whether recent headlines are supportive or harmful,
- **strategy layer** only acts when both agree.

---

## 3. Repository structure

```text
llm_trading_agent_project/
├── llm_trading_agent/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── pipeline.py
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── backtester.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── market_data.py
│   │   └── news_data.py
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   └── paper_broker.py
│   │
│   ├── sentiment/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── finbert_agent.py
│   │   └── openai_agent.py
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── strategy.py
│   │   └── technical.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
│
├── scripts/
│   ├── run_live_signal.py
│   ├── run_paper_trade.py
│   └── run_backtest.py
│
├── artifacts/                # created automatically at runtime
├── requirements.txt
└── README.md
```

---

## 4. What each file does

### `llm_trading_agent/config.py`
This is the central configuration file for the entire project.

It contains dataclasses for:
- `AlpacaConfig`
- `OpenAIConfig`
- `DataConfig`
- `StrategyConfig`
- `SentimentConfig`
- `AppConfig`

It also defines:
- `BASE_DIR`
- `DATA_DIR`
- `DEFAULT_CONFIG`

This file is the easiest place to change the system behavior without touching business logic.

Examples of settings controlled here:
- which stock to trade,
- whether to use the local model or OpenAI,
- SMA window length,
- sentiment thresholds,
- stop-loss and take-profit percentages,
- initial cash for backtesting,
- benchmark symbol,
- how many headlines to fetch.

---

### `llm_trading_agent/models.py`
Defines the project’s internal data structures using dataclasses.

Key objects:
- `NewsItem`: one headline or article summary
- `SentimentRecord`: one model judgment for one headline
- `SignalRecord`: the final signal snapshot used for trading
- `TradeDecision`: the order instructions sent to the broker

These classes make the project easier to maintain because each stage passes structured objects forward rather than loose dictionaries everywhere.

---

### `llm_trading_agent/data/market_data.py`
Contains the `MarketDataHandler` class.

Responsibilities:
- download OHLCV data with `yfinance`,
- normalize columns to lowercase,
- verify required columns exist,
- compute raw daily returns,
- return clean price data for the rest of the system.

Methods:
- `fetch_price_history(symbol, period=None, interval=None)`
- `fetch_benchmark_history()`

Expected output columns:
- `open`
- `high`
- `low`
- `close`
- `volume`
- `return`

---

### `llm_trading_agent/data/news_data.py`
Contains two different news fetchers.

#### `NewsFetcher`
Default news source used by the pipeline.

It:
- queries Google News RSS for a stock symbol,
- parses the feed with `feedparser`,
- filters out older articles using `news_days`,
- keeps only the most recent `news_limit` headlines,
- returns a list of `NewsItem` objects.

#### `AlpacaNewsFetcher`
Optional alternative that pulls news directly from Alpaca’s market data API.

This is included for extensibility and assignment alignment, even though the default pipeline currently uses RSS.

---

### `llm_trading_agent/sentiment/base.py`
Defines the abstract base class `BaseSentimentAgent`.

Any sentiment engine must implement:

```python
analyze(symbol: str, news_items: list[NewsItem]) -> list[SentimentRecord]
```

This design makes it easy to swap the sentiment layer without changing the rest of the pipeline.

---

### `llm_trading_agent/sentiment/finbert_agent.py`
Implements `LocalFinBERTAgent`.

This is the local finance-specific NLP option.

What it does:
- loads `ProsusAI/finbert` through Hugging Face `pipeline`,
- classifies each headline as positive, negative, or neutral,
- converts the raw class confidence into a `signed_score`,
- maps the absolute signed score into a `conviction_score` on a 0–10 scale.

Important behavior:
- positive label -> positive signed score,
- negative label -> negative signed score,
- neutral label -> 0.

This gives you a numeric representation that the strategy layer can use directly.

---

### `llm_trading_agent/sentiment/openai_agent.py`
Implements `OpenAISentimentAgent`.

This is the OpenAI-powered option.

What it does:
- sends the batch of headlines to OpenAI,
- requests strict JSON output using a schema,
- forces each headline to be labeled as `POSITIVE`, `NEGATIVE`, or `NEUTRAL`,
- requires each item to include:
  - `signed_score`
  - `conviction_score`
  - `rationale`
- converts the result into structured `SentimentRecord` objects.

This is the more explicitly “LLM-driven” version of the project.

---

### `llm_trading_agent/signals/technical.py`
Contains the function `compute_technical_features(price_df, sma_window)`.

It adds:
- `sma`
- `price_above_sma`
- `daily_return`

The current technical filter is intentionally simple:
- if `close > sma`, trend is considered favorable.

This simplicity is useful for explaining the architecture during a class demo.

---

### `llm_trading_agent/signals/strategy.py`
Contains the `TradingStrategy` class.

This is where the final signal is formed.

Responsibilities:
- summarize the sentiment records into an average sentiment and average conviction,
- classify overall direction as positive / negative / neutral,
- combine technical and sentiment evidence,
- output a `SignalRecord` containing the final action.

Live signal rule:
- `BUY` when:
  - price is above SMA,
  - average sentiment is at or above the positive threshold,
  - average conviction is at or above the minimum sentiment score.
- `SELL` only if shorting is enabled and negative conditions align.
- otherwise `HOLD`.

This is the core “decision policy” of the repository.

---

### `llm_trading_agent/execution/paper_broker.py`
Contains the `PaperBroker` class.

Responsibilities:
- connect to Alpaca paper trading,
- translate a signal into an executable trade,
- calculate position size based on buying power,
- attach stop-loss and take-profit rules,
- submit a bracket order.

Key methods:
- `build_trade_decision(signal)`
- `submit_trade(decision)`

Current sizing logic:
- uses `position_size_fraction` of current buying power,
- floors the result to an integer number of shares.

Current risk logic:
- stop loss: `close * (1 - stop_loss_pct)`
- take profit: `close * (1 + take_profit_pct)`

---

### `llm_trading_agent/backtest/backtester.py`
Contains:
- `BacktestResult`
- `SimpleBacktester`

This module is the evaluation layer.

What the backtester does:
- iterates row by row through the prepared feature dataframe,
- buys when trend and sentiment are both positive,
- exits on any of these conditions:
  - stop loss hit,
  - take profit hit,
  - max holding period reached,
  - trend breaks,
  - negative news signal appears,
- tracks equity through time,
- stores trade rows,
- computes summary metrics.

Metrics reported:
- `total_return`
- `max_drawdown`
- `sharpe_ratio`

---

### `llm_trading_agent/pipeline.py`
This is the orchestration layer of the entire project.

The `TradingPipeline` class wires everything together.

In `__post_init__`, it creates:
- `MarketDataHandler`
- `NewsFetcher`
- `TradingStrategy`
- either `OpenAISentimentAgent` or `LocalFinBERTAgent`

It exposes two key workflows:

#### `prepare_live_signal()`
1. fetch market data,
2. compute technical features,
3. fetch recent news,
4. analyze sentiment,
5. generate a live signal,
6. return all intermediate pieces for display or execution.

#### `build_backtest_frame(sentiment_csv=None)`
1. fetch 12 months of prices,
2. compute technical features,
3. merge historical sentiment if a CSV is provided,
4. otherwise fill sentiment as neutral,
5. compute rolling sentiment,
6. derive the `sentiment_signal` column.

This module is the cleanest explanation of how the system operates end to end.

---

### `llm_trading_agent/utils/logging_utils.py`
Provides a shared logger factory.

This ensures each module writes timestamped logs in a consistent format.

---

### `scripts/run_live_signal.py`
The simplest script to demonstrate the project.

It:
- builds the pipeline,
- produces one current signal,
- prints a readable summary,
- prints recent headlines,
- prints detailed sentiment rows.

This is the best script for your class demo if you want to show the model reasoning pipeline before showing execution.

---

### `scripts/run_paper_trade.py`
This script moves from analysis to action.

It:
- builds the same live signal,
- stops if the result is `HOLD`,
- otherwise creates a `PaperBroker`,
- converts the signal into a trade decision,
- submits a paper order to Alpaca.

This is the main execution demo.

---

### `scripts/run_backtest.py`
This is the evaluation script.

It:
- prepares the backtest dataframe,
- runs the strategy through the backtester,
- fetches benchmark data,
- computes benchmark equity,
- compares strategy vs benchmark,
- prints summary metrics,
- saves artifacts to disk.

Artifacts written to `artifacts/`:
- `equity_curve.csv`
- `trades.csv`
- `metrics.csv`
- `equity_curve.png`

---

## 5. Configuration reference

The repository is controlled primarily from `llm_trading_agent/config.py`.

### `AlpacaConfig`
```python
api_key: str
secret_key: str
paper: bool = True
```
Use your Alpaca paper credentials here.

### `OpenAIConfig`
```python
api_key: str
model: str = "gpt-4.1-mini"
```
Used only when sentiment mode is `openai`.

### `DataConfig`
```python
benchmark_symbol: str = "SPY"
lookback_period: str = "18mo"
interval: str = "1d"
news_limit: int = 12
news_days: int = 7
```
Controls how much market data and news are collected.

### `StrategyConfig`
```python
symbol: str = "AAPL"
sma_window: int = 50
sentiment_lookback_days: int = 3
min_sentiment_score: float = 6.5
positive_threshold: float = 0.15
negative_threshold: float = -0.15
position_size_fraction: float = 0.95
stop_loss_pct: float = 0.03
take_profit_pct: float = 0.08
max_holding_days: int = 10
initial_cash: float = 100_000.0
allow_short: bool = False
```
This is the most important block for strategy behavior.

### `SentimentConfig`
```python
mode: Literal["local", "openai"] = "local"
local_model_name: str = "ProsusAI/finbert"
max_headlines_per_call: int = 8
```
Change `mode` to switch between the local model and OpenAI.

---

## 6. Prerequisites

You need:
- Python 3.10+ recommended,
- internet access for market data and news,
- an Alpaca paper trading account if you want to submit orders,
- an OpenAI API key only if you choose OpenAI mode.

Note that the local FinBERT model will download model weights on first run, so the first launch may take longer.

---

## 7. Installation

From the project root, install the dependencies:

```bash
pip install -r requirements.txt
```

If your system uses `python3` and `pip3`, use:

```bash
pip3 install -r requirements.txt
```

---

## 7.5 How to Obtain API Keys

Before running the project, you must obtain credentials for OpenAI and Alpaca Paper Trading. These services power the sentiment analysis and paper trading components of the system.

⚠️ Never commit your API keys to GitHub.

OpenAI API Key (LLM Sentiment Option)

OpenAI is used when the project runs in LLM sentiment mode.

The OpenAI model interprets the news headlines and produces structured sentiment outputs that the trading strategy can use.

Steps to obtain the key

Go to:

https://platform.openai.com

Create an account or sign in.

Navigate to:

Dashboard → API Keys

Click:

Create new secret key

You will receive a key similar to:

sk-xxxxxxxxxxxxxxxxxxxxxxxx

This key allows the application to send requests to the OpenAI API.

Alpaca Paper Trading API

Alpaca provides a simulated trading environment that allows the system to place orders without using real money.

The project uses Alpaca for:

retrieving account information

determining buying power

submitting bracket orders

managing stop-loss and take-profit rules

Steps to obtain Alpaca keys

Go to:

https://alpaca.markets

Create a free account.

Log into the trading dashboard:

https://app.alpaca.markets

Navigate to:

Paper Trading → API Keys

Click:

Generate New Key

You will receive two values:

API Key ID
Secret Key

Example format:

API Key ID: PKXXXXXXXXXXXX
Secret Key: xxxxxxxxxxxxxxxxxxxxx

The paper trading endpoint used by this project is:

https://paper-api.alpaca.markets

---

## 8. Where to put your API keys

Edit `llm_trading_agent/config.py` and replace the placeholders.

Example:

```python
@dataclass(frozen=True)
class AlpacaConfig:
    api_key: str = "YOUR_ALPACA_PAPER_KEY"
    secret_key: str = "YOUR_ALPACA_PAPER_SECRET"
    paper: bool = True


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str = "YOUR_OPENAI_API_KEY"
    model: str = "gpt-4.1-mini"
```

If you want to use OpenAI sentiment, also set:

```python
@dataclass(frozen=True)
class SentimentConfig:
    mode: Literal["local", "openai"] = "openai"
```

If you want to stay fully local, leave:

```python
mode = "local"
```

---

## 9. How to run the project

# Optional: Quick Launcher Scripts

For convenience, the repository includes two launcher scripts that provide a simple interactive menu for running the main workflows (file end with .bat or .sh).

These scripts are optional. The project can always be run directly using Python commands.

Always run commands from the project root directory.

### A. Generate a live signal

```bash
python scripts/run_live_signal.py
```

What this script does:
1. downloads recent price history for the configured symbol,
2. computes SMA features,
3. fetches recent news,
4. classifies sentiment,
5. prints the final action and supporting context.

Typical output includes:
- symbol,
- action,
- close,
- SMA,
- average sentiment,
- conviction,
- reason,
- headlines,
- sentiment records.

---

### B. Submit a paper trade

```bash
python scripts/run_paper_trade.py
```

What this script does:
1. runs the same live signal pipeline,
2. checks whether the action is `BUY`, `SELL`, or `HOLD`,
3. exits early if `HOLD`,
4. builds a bracket order if tradable,
5. sends the order to Alpaca paper trading.

Use this script only after you have confirmed:
- your Alpaca key is correct,
- the market is open or your order type behavior is acceptable for your demo,
- you are comfortable submitting a simulated order.

---

### C. Run the backtest without historical sentiment

```bash
python scripts/run_backtest.py
```

This is the fastest demo run, but there is an important limitation:
- if you do not provide a historical sentiment CSV, the code fills sentiment with `0.0`,
- so the backtest becomes closer to a trend-filter skeleton than a full historical LLM backtest.

It is still useful for showing pipeline structure and benchmark comparison, but it is not the strongest empirical validation.

---

### D. Run the backtest with historical sentiment

```bash
python scripts/run_backtest.py --sentiment-csv your_sentiment_history.csv
```

Expected CSV format:

```csv
date,signed_score
2025-05-01,0.42
2025-05-02,-0.11
2025-05-03,0.20
```

Behavior:
- the script groups sentiment by date,
- merges it into the price dataframe,
- computes rolling sentiment,
- converts that into `sentiment_signal`.

This is the preferred method if you want the backtest to reflect a more realistic historical sentiment process.

---

## 10. Walkthrough of the live pipeline

This section explains exactly what happens when `run_live_signal.py` is executed.

### Step 1: Load configuration
The script imports `DEFAULT_CONFIG` from `config.py`.

This determines:
- which stock is analyzed,
- which sentiment engine is used,
- what the SMA window is,
- how much news to fetch,
- what thresholds define a valid trade.

### Step 2: Initialize `TradingPipeline`
The pipeline constructor creates:
- `MarketDataHandler`
- `NewsFetcher`
- `TradingStrategy`
- either `LocalFinBERTAgent` or `OpenAISentimentAgent`

This means the script itself stays simple while the pipeline handles orchestration.

### Step 3: Download market data
`MarketDataHandler.fetch_price_history(symbol)` calls Yahoo Finance through `yfinance`.

Returned data is cleaned and standardized into a dataframe.

### Step 4: Compute technical features
`compute_technical_features()` adds:
- rolling simple moving average,
- price-above-SMA flag,
- daily return.

The current live signal ultimately uses the latest row of this enriched dataframe.

### Step 5: Fetch recent news
`NewsFetcher.fetch(symbol)` pulls recent headlines from Google News RSS.

Only recent items within the `news_days` cutoff are kept.

### Step 6: Run sentiment analysis
The selected sentiment agent processes the headlines.

#### If using FinBERT:
Each headline receives:
- label,
- raw model confidence,
- signed score,
- conviction score.

#### If using OpenAI:
The entire headline batch is sent with a strict JSON schema and each item is returned with:
- label,
- signed score,
- conviction score,
- rationale.

### Step 7: Summarize sentiment
`TradingStrategy.summarize_sentiment()` computes:
- average signed sentiment,
- average conviction,
- overall direction.

### Step 8: Generate the action
`TradingStrategy.generate_live_signal()` compares:
- current close vs SMA,
- average sentiment vs threshold,
- average conviction vs minimum score.

Then it produces one of:
- `BUY`
- `SELL`
- `HOLD`

### Step 9: Print an explainable result
The script prints both the final signal and the intermediate evidence so the output is presentation-friendly.

That makes it easy to explain:
- what news was seen,
- how the model interpreted it,
- why the final action happened.

---

## 11. Walkthrough of the execution pipeline

This section explains `run_paper_trade.py`.

### Step 1: Reuse the live signal pipeline
The paper trade script first performs the same analysis as the live signal script.

This is good architecture because execution is only a downstream consumer of the signal, not mixed into the analysis stage.

### Step 2: Exit if there is no actionable setup
If the strategy says `HOLD`, the script prints a message and stops.

That prevents unnecessary paper orders.

### Step 3: Build a trade decision
`PaperBroker.build_trade_decision(signal)`:
- requests account details from Alpaca,
- reads current buying power,
- allocates a fraction of buying power,
- computes integer share quantity,
- computes stop-loss and take-profit levels.

### Step 4: Submit bracket order
`PaperBroker.submit_trade(decision)` builds a market order with:
- side,
- quantity,
- bracket structure,
- stop-loss child order,
- take-profit child order.

This is important for the rubric because risk controls are attached programmatically rather than handled manually.

---

## 12. Walkthrough of the backtest pipeline

This section explains `run_backtest.py`.

### Step 1: Build the backtest dataframe
The script calls `pipeline.build_backtest_frame(sentiment_csv)`.

That function:
1. downloads 12 months of price history,
2. computes technical features,
3. merges historical sentiment if provided,
4. fills missing sentiment with neutral values,
5. computes rolling average sentiment,
6. derives `sentiment_signal`.

### Step 2: Run the rule-based simulation
`SimpleBacktester.run(df)` then iterates over the data chronologically.

The strategy can enter when:
- no shares are currently held,
- price is above SMA,
- sentiment signal is positive.

The strategy can exit when any of the following occur:
- stop-loss threshold reached,
- take-profit threshold reached,
- holding period timeout,
- price falls back below SMA,
- negative sentiment appears.

### Step 3: Track equity and trades
For each row, the backtester stores:
- portfolio equity,
- remaining cash,
- current share count,
- close price.

Trade events are also recorded as a separate table.

### Step 4: Compute summary metrics
The backtester calculates:
- total return,
- maximum drawdown,
- Sharpe ratio.

### Step 5: Compare against benchmark
The script also downloads benchmark data, builds a buy-and-hold equity curve, and prints both strategies side by side.

### Step 6: Save artifacts
The script saves CSV files and a PNG plot into `artifacts/`.

These files are ideal for:
- your report,
- your presentation slides,
- your live class demo.

---

## 13. Artifacts and outputs

After running the backtest, the `artifacts/` folder should contain:

### `equity_curve.csv`
Portfolio value through time.

### `trades.csv`
Every simulated trade event.

### `metrics.csv`
Strategy and benchmark summary metrics.

### `equity_curve.png`
A chart comparing strategy equity vs benchmark buy-and-hold.

---

## 14. Example of a historical sentiment CSV

If you want the backtest to use historical sentiment, create a file like this:

```csv
date,signed_score
2025-01-03,0.35
2025-01-06,0.22
2025-01-07,-0.18
2025-01-08,0.40
2025-01-09,0.11
```

Interpretation:
- each row is the average or representative signed sentiment for that date,
- the code will group by date before merging,
- missing dates are filled with neutral sentiment.

A stronger extension would be to save headline-level sentiment daily and aggregate automatically.

---

## 15. Recommended demo order for class

A strong 10-minute live presentation could follow this structure:

1. **Problem framing**  
   Explain that pure price prediction with an LLM is weak, so the model is only used on unstructured text.

2. **Architecture**  
   Show the repo structure and the flow: data -> sentiment -> strategy -> execution -> validation.

3. **Live signal demo**  
   Run `python scripts/run_live_signal.py`.

4. **Explain the evidence**  
   Show headlines, sentiment labels, average sentiment, and the technical filter.

5. **Execution demo**  
   Run `python scripts/run_paper_trade.py` if the action is tradable.

6. **Backtest summary**  
   Run `python scripts/run_backtest.py` and show the output files in `artifacts/`.

7. **Discuss one limitation honestly**  
   Mention that the strongest backtest requires historical sentiment collection rather than neutral placeholders.

That last point is helpful because it shows maturity and awareness rather than overclaiming.

---

## 16. Strengths of the current implementation

1. **Modular design**  
   Responsibilities are cleanly separated.

2. **Clear role for the LLM**  
   The language model is used for text interpretation, not price arithmetic.

3. **Explainable signal generation**  
   Trade decisions are based on transparent thresholds.

4. **Built-in risk controls**  
   Stop-loss and take-profit are attached to execution.

5. **Reproducible workflow**  
   Scripts can be run independently for signal generation, execution, and backtesting.

6. **Easy to extend**  
   The abstract sentiment interface and pipeline structure make future changes straightforward.

---

## 17. Current limitations

This repository is a strong assignment prototype, but it is still intentionally simple. Main limitations include:

1. **Historical sentiment is optional, not automatically built**  
   Without a sentiment CSV, the backtest is only a partial demonstration.

2. **Single-symbol focus**  
   The system currently analyzes one configured ticker at a time.

3. **Simple technical filter**  
   Only an SMA trend check is used.

4. **No transaction costs or slippage model**  
   The backtester does not account for commissions, spread, or execution delay.

5. **No portfolio-level allocation logic**  
   Position sizing is based only on available buying power.

6. **No persistent database or cache**  
   News and results are not stored in a long-term research store.

These limitations are acceptable in a course project as long as they are acknowledged clearly.

---

## 18. Suggested future improvements

If you continue this repository, the next most valuable improvements would be:

- add multi-ticker ranking and selection,
- store daily news and sentiment into a CSV or database,
- add retry logic and API error handling,
- support Alpaca News as the default news source,
- add more technical filters such as momentum or volatility,
- include transaction cost modeling,
- build a small dashboard for demo purposes,
- generate a full report notebook automatically.

---

## 19. Troubleshooting

### Problem: `No market data returned`
Likely causes:
- invalid symbol,
- internet issue,
- Yahoo Finance temporary response issue.

Try another symbol such as `AAPL` or `MSFT`.

### Problem: no recent news found
Likely causes:
- the query returned too little recent news,
- the `news_days` filter is too strict.

Try increasing `news_days` in `DataConfig`.

### Problem: OpenAI call fails
Check:
- API key is correct,
- billing or quota is available,
- `SentimentConfig.mode` is set correctly.

### Problem: Alpaca order fails
Check:
- key and secret are for the paper account,
- `paper=True`,
- market state and symbol are valid,
- quantity is not zero.

### Problem: FinBERT takes time on first launch
This is normal. The model is downloaded on first use.

---

## 20. Minimal quick start

If you only need the shortest possible setup:

1. install dependencies
```bash
pip install -r requirements.txt
```

2. add your keys to `llm_trading_agent/config.py`

3. choose sentiment mode in `config.py`

4. run a live signal
```bash
python scripts/run_live_signal.py
```

5. run a backtest
```bash
python scripts/run_backtest.py
```

6. optionally run paper execution
```bash
python scripts/run_paper_trade.py
```

---

## 21. Summary

This repository implements a maintainable and presentation-friendly prototype of an LLM-driven trading agent.

Its key idea is simple and defensible:
- use language intelligence for **news interpretation**,
- use deterministic rules for **signal control**,
- use programmatic bracket orders for **risk-managed execution**,
- and use backtesting plus benchmark comparison for **validation**.

That combination makes the project substantially stronger than a single-file demo and gives a clear story for both code quality and model purpose.
