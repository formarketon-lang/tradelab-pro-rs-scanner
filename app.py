import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

# =========================================================
# TradeLab Pro: RS + Scanner + Backtesting + Dashboard
# =========================================================

st.set_page_config(
    page_title="TradeLab Pro",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: #111827;
}
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    padding: 12px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 TradeLab Pro")
st.caption(
    "Relative Strength • Cash Market Scanner • Backtesting • Dashboard"
)

# -------------------- Data normalization --------------------

def normalize(df):
    """Normalize Yahoo Finance or CSV OHLCV data."""

    if df is None or df.empty:
        raise ValueError("কোনো ডেটা পাওয়া যায়নি।")

    df = df.copy()

    # Handle Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        selected_level = None

        for level in range(df.columns.nlevels):
            values = {
                str(v).strip().lower()
                for v in df.columns.get_level_values(level)
            }
            if "close" in values:
                selected_level = level
                break

        if selected_level is None:
            raise ValueError(
                "Yahoo Finance columns-এ Close field পাওয়া যায়নি।"
            )

        df.columns = df.columns.get_level_values(selected_level)

    df.columns = [str(c).strip() for c in df.columns]

    # Move date index into a column
    if not any(str(c).lower() in ("date", "datetime", "timestamp", "time")
               for c in df.columns):
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "Date"})
        else:
            raise ValueError(
                "Date কলাম বা DatetimeIndex পাওয়া যায়নি।"
            )

    rename = {}

    for col in df.columns:
        key = str(col).strip().lower().replace("_", " ")

        if key in ("date", "datetime", "timestamp", "time"):
            rename[col] = "Date"
        elif key in ("open", "o"):
            rename[col] = "Open"
        elif key in ("high", "h"):
            rename[col] = "High"
        elif key in ("low", "l"):
            rename[col] = "Low"
        elif key in ("close", "c"):
            rename[col] = "Close"
        elif key in ("adj close", "adjusted close", "adjclose"):
            # Use adjusted close only if ordinary Close is absent
            if "Close" not in rename.values():
                rename[col] = "Close"
        elif key in ("volume", "vol", "v"):
            rename[col] = "Volume"
        elif key in ("symbol", "ticker", "scrip"):
            rename[col] = "Symbol"

    df = df.rename(columns=rename)

    if "Date" not in df.columns:
        raise ValueError("Date কলাম পাওয়া যায়নি।")

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    df = df.dropna(subset=["Date"]).copy()

    # Keep only the fields needed for OHLCV calculations
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            if col == "Volume":
                df[col] = 0
            else:
                raise ValueError(f"প্রয়োজনীয় কলাম নেই: {col}")

        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    df = df[
        (df["Close"] > 0) &
        (df["High"] >= df["Low"]) &
        (df["Open"] > 0)
    ].copy()

    df = (
        df.drop_duplicates("Date", keep="last")
        .sort_values("Date")
        .set_index("Date")
    )

    if df.empty:
        raise ValueError("যাচাইয়ের পরে কোনো valid OHLC data নেই।")

    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


@st.cache_data(ttl=900, show_spinner=False)
def yahoo(ticker, start, end):
    """Download historical data from Yahoo Finance."""

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=25
    )

    if data is None or data.empty:
        raise ValueError(
            f"{ticker}: Yahoo Finance ডেটা পাওয়া যায়নি। "
            "Ticker, date range বা network যাচাই করুন।"
        )

    return normalize(data)


def load_pair(
    symbol, source, stock_csv, bench_csv,
    csv_symbol, start, end, benchmark
):
    """Load stock and benchmark data."""

    if source == "Yahoo Finance":
        stock = yahoo(symbol, str(start), str(end))
        bench = yahoo(benchmark, str(start), str(end))

    else:
        if stock_csv is None or bench_csv is None:
            raise ValueError(
                "CSV mode-এ Stock এবং Benchmark—দুটি CSV-ই আপলোড করুন।"
            )

        stock_file = pd.read_csv(stock_csv)
        bench_file = pd.read_csv(bench_csv)

        if "Symbol" in stock_file.columns:
            if not csv_symbol:
                raise ValueError("Stock CSV থেকে একটি Symbol নির্বাচন করুন।")

            stock_file = stock_file[
                stock_file["Symbol"].astype(str).str.upper()
                == str(csv_symbol).upper()
            ].drop(columns=["Symbol"])

        stock = normalize(stock_file)
        bench = normalize(bench_file)

    stock = stock.loc[str(start):str(end)]
    bench = bench.loc[str(start):str(end)]

    if stock.empty or bench.empty:
        raise ValueError("নির্বাচিত তারিখে পর্যাপ্ত ডেটা নেই।")

    return stock, bench


# -------------------- Core calculations --------------------

def ret(series, n):
    if len(series) <= n:
        return np.nan

    previous = series.iloc[-n - 1]
    latest = series.iloc[-1]

    if previous == 0:
        return np.nan

    return (latest / previous - 1) * 100


def joined_data(stock, bench):
    return pd.concat(
        [
            stock["Close"].rename("Stock"),
            bench["Close"].rename("Benchmark")
        ],
        axis=1
    ).dropna()


def rs_stats(data, periods):
    rows = []

    for n in periods:
        stock_return = ret(data["Stock"], n)
        bench_return = ret(data["Benchmark"], n)

        if pd.notna(stock_return) and pd.notna(bench_return):
            excess = stock_return - bench_return

            if bench_return != -100:
                ratio = (
                    (1 + stock_return / 100) /
                    (1 + bench_return / 100)
                )
            else:
                ratio = np.nan
        else:
            excess = np.nan
            ratio = np.nan

        rows.append({
            "Window (trading days)": n,
            "Stock Return %": stock_return,
            "Benchmark Return %": bench_return,
            "Excess Return (pp)": excess,
            "Relative Ratio": ratio,
            "Common Dates": len(data)
        })

    return pd.DataFrame(rows)


def indicators(stock):
    result = stock.copy()
    result["SMA50"] = result["Close"].rolling(50).mean()
    result["AvgVol20"] = result["Volume"].rolling(20).mean()
    result["High20Prev"] = result["High"].shift(1).rolling(20).max()
    return result


def backtest(stock, bench, rs_window, sma, capital, cost_bps):
    data = joined_data(stock, bench).copy()

    if len(data) < max(rs_window, sma) + 5:
        raise ValueError(
            "Backtest-এর জন্য আরও বেশি historical data প্রয়োজন।"
        )

    data["stock_ret"] = data["Stock"].pct_change().fillna(0)
    data["rs"] = data["Stock"] / data["Benchmark"]
    data["rs_ma"] = data["rs"].rolling(rs_window).mean()
    data["price_ma"] = data["Stock"].rolling(sma).mean()

    # Signal calculated from available closing data.
    signal = (
        (data["rs"] > data["rs_ma"]) &
        (data["Stock"] > data["price_ma"])
    ).astype(int)

    # Execute on next bar to reduce look-ahead bias.
    data["position"] = signal.shift(1).fillna(0)
    data["turnover"] = (
        data["position"].diff().abs()
        .fillna(data["position"].abs())
    )

    data["strategy_ret"] = (
        data["position"] * data["stock_ret"]
        - data["turnover"] * (cost_bps / 10000)
    )

    data["equity"] = capital * (
        1 + data["strategy_ret"]
    ).cumprod()

    data["buy_hold"] = capital * (
        1 + data["stock_ret"]
    ).cumprod()

    data["drawdown"] = (
        data["equity"] / data["equity"].cummax() - 1
    )

    daily = data["strategy_ret"]
    daily_std = daily.std()

    metrics = {
        "Strategy return %":
            (data["equity"].iloc[-1] / capital - 1) * 100,
        "Buy & hold return %":
            (data["buy_hold"].iloc[-1] / capital - 1) * 100,
        "Max drawdown %": data["drawdown"].min() * 100,
        "Annualized volatility %":
            daily_std * np.sqrt(252) * 100,
        "Sharpe (risk-free 0)":
            daily.mean() / daily_std * np.sqrt(252)
            if pd.notna(daily_std) and daily_std > 0 else np.nan,
        "Position changes": int((data["turnover"] > 0).sum()),
        "Observations": len(data)
    }

    return data, metrics


# -------------------- Sidebar settings --------------------

with st.sidebar:
    st.header("⚙️ Data Settings")

    source = st.radio(
        "Data source",
        ["Yahoo Finance", "CSV upload"]
    )

    exchange = st.selectbox(
        "Exchange",
        ["NSE (.NS)", "BSE (.BO)"]
    )

    today = date.today()
    default_start = today - timedelta(days=730)

    start = st.date_input(
        "Start date",
        value=default_start,
        max_value=today
    )

    end = st.date_input(
        "End date (exclusive)",
        value=today + timedelta(days=1),
        min_value=start + timedelta(days=1)
    )

    default_benchmark = (
        "^NSEI" if exchange.startswith("NSE") else "^BSESN"
    )

    benchmark = st.text_input(
        "Benchmark ticker",
        default_benchmark
    ).strip().upper()

    stock_csv = None
    bench_csv = None
    csv_symbol = None

    if source == "CSV upload":
        stock_csv = st.file_uploader(
            "Stock OHLCV CSV",
            type=["csv"],
            key="stock_upload"
        )

        bench_csv = st.file_uploader(
            "Benchmark OHLCV CSV",
            type=["csv"],
            key="bench_upload"
        )

        if stock_csv is not None:
            try:
                stock_csv.seek(0)
                preview = pd.read_csv(stock_csv)

                if "Symbol" in preview.columns:
                    symbols = sorted(
                        preview["Symbol"]
                        .dropna()
                        .astype(str)
                        .unique()
                    )

                    if symbols:
                        csv_symbol = st.selectbox(
                            "Stock CSV symbol",
                            symbols
                        )

                stock_csv.seek(0)

            except Exception as exc:
                st.error(f"CSV পড়তে সমস্যা: {exc}")

    st.caption(
        "Yahoo Finance একটি convenience research source; "
        "এটি guaranteed live বা exchange-certified feed নয়।"
    )


# -------------------- Tabs --------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Relative Strength",
    "Scanner",
    "Backtesting",
    "Data Quality"
])


# ==================== TAB 1: DASHBOARD ====================

with tab1:
    st.subheader("📈 Dashboard")

    symbol = st.text_input(
        "Stock ticker",
        "RELIANCE.NS",
        key="dashboard_symbol"
    ).strip().upper()

    if st.button("Load Dashboard", key="load_dashboard"):
        try:
            stock, bench = load_pair(
                symbol, source, stock_csv, bench_csv,
                csv_symbol, start, end, benchmark
            )

            data = joined_data(stock, bench)

            if data.empty:
                st.error("Stock ও benchmark-এর common dates নেই.")
            else:
                stock_return = ret(data["Stock"], 20)
                bench_return = ret(data["Benchmark"], 20)

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Latest Close",
                    f"₹{data['Stock'].iloc[-1]:,.2f}"
                )

                c2.metric(
                    "Stock 20D Return",
                    f"{stock_return:.2f}%"
                    if pd.notna(stock_return) else "N/A"
                )

                c3.metric(
                    "20D Excess Return",
                    f"{stock_return - bench_return:.2f} pp"
                    if pd.notna(stock_return) and pd.notna(bench_return)
                    else "N/A"
                )

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Stock"],
                        mode="lines",
                        name="Stock Close"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=400,
                    title=f"{symbol} Historical Close",
                    xaxis_title="Date",
                    yaxis_title="Price"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(data.tail(10), use_container_width=True)

        except Exception as exc:
            st.error(f"Dashboard error: {exc}")


# ==================== TAB 2: RELATIVE STRENGTH ====================

with tab2:
    st.subheader("💪 Relative Strength Analysis")

    symbol = st.text_input(
        "Stock ticker",
        "HINDUNILVR.NS",
        key="rs_symbol"
    ).strip().upper()

    periods = st.multiselect(
        "Return windows (trading days)",
        [5, 10, 20, 60, 120, 252],
        default=[20, 60, 120]
    )

    if st.button("Calculate RS", key="calculate_rs"):
        try:
            stock, bench = load_pair(
                symbol, source, stock_csv, bench_csv,
                csv_symbol, start, end, benchmark
            )

            data = joined_data(stock, bench)

            if len(data) < 2:
                st.error("Common trading data অপর্যাপ্ত।")
            else:
                result = rs_stats(data, periods or [20])
                st.dataframe(
                    result.round(4),
                    use_container_width=True
                )

                ratio = data["Stock"] / data["Benchmark"]
                normalized_rs = ratio / ratio.iloc[0] * 100

                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Stock"],
                        name="Stock Close"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data["Benchmark"],
                        name="Benchmark Close"
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=normalized_rs,
                        name="Normalized RS (start=100)",
                        yaxis="y2"
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=450,
                    title="Stock vs Benchmark",
                    yaxis=dict(title="Price / Index"),
                    yaxis2=dict(
                        title="Normalized RS",
                        overlaying="y",
                        side="right"
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Excess Return = Stock Return − Benchmark Return "
                    "(percentage points)। Relative Strength এখানে "
                    "RSI নয় এবং একা Buy/Sell signal নয়।"
                )

        except Exception as exc:
            st.error(f"Relative Strength error: {exc}")


# ==================== TAB 3: SCANNER ====================

with tab3:
    st.subheader("🔎 Cash Market Scanner")

    st.caption(
        "এটি manually selected ticker universe scan করে। "
        "এটি সম্পূর্ণ NSE/BSE symbol master নয়।"
    )

    raw_symbols = st.text_area(
        "Symbols — comma-separated",
        "RELIANCE,HINDUNILVR,ICICIBANK,INFY,TCS"
    )

    col1, col2, col3 = st.columns(3)

    trend_on = col1.checkbox("Close > SMA50", value=True)
    rs_on = col2.checkbox("20D Excess Return threshold", value=True)
    volume_on = col3.checkbox("Volume > 20D average", value=False)

    threshold = st.number_input(
        "Minimum excess return (percentage points)",
        min_value=-100.0,
        max_value=100.0,
        value=0.0,
        step=0.5
    )

    limit = st.slider(
        "Maximum symbols",
        min_value=1,
        max_value=30,
        value=10
    )

    if st.button("Run Scanner", key="run_scanner"):
        if source != "Yahoo Finance":
            st.warning(
                "Multi-symbol scanner বর্তমানে Yahoo Finance mode-এ চলে। "
                "RS ও Backtesting-এর জন্য CSV mode ব্যবহার করতে পারেন।"
            )
        else:
            symbols = [
                item.strip().upper()
                for item in raw_symbols.replace("\n", ",").split(",")
                if item.strip()
            ][:limit]

            if exchange.startswith("NSE"):
                symbols = [
                    item if "." in item or item.startswith("^")
                    else item + ".NS"
                    for item in symbols
                ]
            else:
                symbols = [
                    item if "." in item or item.startswith("^")
                    else item + ".BO"
                    for item in symbols
                ]

            rows = []
            errors = []

            try:
                bench = yahoo(benchmark, str(start), str(end))
            except Exception as exc:
                bench = None
                st.error(f"Benchmark error: {exc}")

            if bench is not None and symbols:
                progress = st.progress(0)

                for i, ticker in enumerate(symbols):
                    try:
                        stock = yahoo(ticker, str(start), str(end))
                        data = joined_data(stock, bench)
                        ind = indicators(stock)

                        if len(data) < 55:
                            raise ValueError(
                                "প্রায় 55 বা তার বেশি common "
                                "trading observations প্রয়োজন।"
                            )

                        stock_return = ret(data["Stock"], 20)
                        bench_return = ret(data["Benchmark"], 20)

                        excess = (
                            stock_return - bench_return
                            if pd.notna(stock_return)
                            and pd.notna(bench_return)
                            else np.nan
                        )

                        latest = ind.iloc[-1]

                        trend_pass = bool(
                            pd.notna(latest["SMA50"])
                            and latest["Close"] > latest["SMA50"]
                        )

                        volume_pass = bool(
                            pd.notna(latest["AvgVol20"])
                            and latest["Volume"] > latest["AvgVol20"]
                        )

                        checks = []

                        if trend_o
