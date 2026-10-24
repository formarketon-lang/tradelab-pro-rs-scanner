import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(
    page_title="TradeLab Pro",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.stApp {background:#0b1120;color:#e2e8f0}
[data-testid="stSidebar"] {background:#111827}
[data-testid="stMetric"] {
 background:#1e293b;border:1px solid #334155;
 padding:12px;border-radius:12px
}
</style>
""", unsafe_allow_html=True)

st.title("TradeLab Pro — RS + Scanner + Backtesting")
st.caption("Indian cash equities | Historical research | No order execution")


# --------------------------------------------------
# 1. DATA NORMALIZATION
# --------------------------------------------------

def normalize(df):
    df = df.copy()

    if df.empty:
        raise ValueError("কোনো ডেটা পাওয়া যায়নি।")

    # Handle Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        fields = {
            "date", "datetime", "timestamp", "open",
            "high", "low", "close", "adj close",
            "adjclose", "volume", "symbol", "ticker"
        }

        chosen_level = None
        for level in range(df.columns.nlevels):
            vals = {
                str(x).strip().lower()
                for x in df.columns.get_level_values(level)
            }
            if vals & fields:
                chosen_level = level
                break

        if chosen_level is None:
            raise ValueError("ডেটার কলাম শনাক্ত করা যায়নি।")

        df.columns = df.columns.get_level_values(chosen_level)

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    rename = {}
    for c in df.columns:
        k = c.lower().replace("_", " ").strip()

        if k in ("date", "datetime", "timestamp", "time", "index"):
            rename[c] = "Date"
        elif k in ("open", "o"):
            rename[c] = "Open"
        elif k in ("high", "h"):
            rename[c] = "High"
        elif k in ("low", "l"):
            rename[c] = "Low"
        elif k in ("close", "c", "adj close", "adjclose",
                   "adjusted close"):
            if "Close" not in rename.values():
                rename[c] = "Close"
        elif k in ("volume", "vol", "v"):
            rename[c] = "Volume"
        elif k in ("symbol", "ticker", "scrip"):
            rename[c] = "Symbol"

    df = df.rename(columns=rename)

    if "Date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df.index.name = "Date"
            df = df.reset_index()
        else:
            raise ValueError("Date কলাম পাওয়া যায়নি।")

    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce", utc=True
    ).dt.tz_convert(None)

    df = df.dropna(subset=["Date"]).sort_values("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c not in df.columns:
            if c == "Volume":
                df[c] = 0.0
            else:
                raise ValueError(f"প্রয়োজনীয় কলাম নেই: {c}")

        col = df[c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]

        df[c] = pd.to_numeric(col, errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[
        (df["Open"] > 0) &
        (df["Close"] > 0) &
        (df["High"] >= df["Low"])
    ]

    df = (
        df.drop_duplicates("Date", keep="last")
          .sort_values("Date")
          .set_index("Date")
    )

    if df.empty:
        raise ValueError("যাচাইয়ের পরে কোনো বৈধ ডেটা নেই।")

    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


# --------------------------------------------------
# 2. YAHOO FINANCE
# --------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def yahoo(ticker, start, end):
    d = yf.download(
        ticker,
        start=str(start),
        end=str(end),
        auto_adjust=False,
        progress=False,
        threads=False,
        group_by="column",
        timeout=20
    )

    if d is None or d.empty:
        raise ValueError(
            f"{ticker}: ডেটা পাওয়া যায়নি। "
            "Ticker, date range বা Yahoo Finance connection পরীক্ষা করুন।"
        )

    return normalize(d)


def load_pair(symbol, source, stock_file, bench_file,
              csv_symbol, start, end, benchmark):

    if source == "Yahoo Finance":
        stock = yahoo(symbol, start, end)
        bench = yahoo(benchmark, start, end)
    else:
        if stock_file is None or bench_file is None:
            raise ValueError("Stock এবং Benchmark—দুটি CSV আপলোড করুন।")

        stock_file.seek(0)
        bench_file.seek(0)
        sf = pd.read_csv(stock_file)
        bf = pd.read_csv(bench_file)

        if "Symbol" in sf.columns:
            if not csv_symbol:
                raise ValueError("Stock CSV থেকে Symbol নির্বাচন করুন.")
            sf = sf[
                sf["Symbol"].astype(str).str.upper()
                == str(csv_symbol).upper()
            ].drop(columns=["Symbol"])

        stock = normalize(sf)
        bench = normalize(bf)

    stock = stock.loc[
        (stock.index >= pd.Timestamp(start)) &
        (stock.index < pd.Timestamp(end))
    ]
    bench = bench.loc[
        (bench.index >= pd.Timestamp(start)) &
        (bench.index < pd.Timestamp(end))
    ]

    return stock, bench


# --------------------------------------------------
# 3. CALCULATIONS
# --------------------------------------------------

def ret(series, n):
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= n:
        return np.nan

    first = float(series.iloc[-n-1])
    last = float(series.iloc[-1])

    if first == 0:
        return np.nan

    return (last / first - 1) * 100


def joined_data(stock, bench):
    return pd.concat(
        [
            stock["Close"].rename("Stock"),
            bench["Close"].rename("Benchmark")
        ],
        axis=1,
        join="inner"
    ).dropna().sort_index()


def rs_stats(j, periods):
    rows = []

    for n in periods:
        sr = ret(j["Stock"], n)
        br = ret(j["Benchmark"], n)

        rows.append({
            "Window (trading days)": n,
            "Stock Return %": sr,
            "Benchmark Return %": br,
            "Excess Return (pp)": (
                sr - br if pd.notna(sr) and pd.notna(br) else np.nan
            ),
            "Relative Ratio": (
                (1 + sr / 100) / (1 + br / 100)
                if pd.notna(sr) and pd.notna(br) and br != -100
                else np.nan
            ),
            "Common dates": len(j)
        })

    return pd.DataFrame(rows)


def indicators(stock):
    x = stock.copy()
    x["SMA50"] = x["Close"].rolling(50).mean()
    x["AvgVol20"] = x["Volume"].rolling(20).mean()
    x["High20Prev"] = x["High"].shift(1).rolling(20).max()
    return x


# --------------------------------------------------
# 4. BACKTEST
# --------------------------------------------------

def backtest(stock, bench, rs_window, sma, capital, cost_bps):
    j = joined_data(stock, bench).copy()

    if len(j) < max(rs_window, sma) + 2:
        raise ValueError("Backtest-এর জন্য আরও বেশি historical data দরকার।")

    j["stock_ret"] = j["Stock"].pct_change().fillna(0)
    j["rs"] = j["Stock"] / j["Benchmark"]
    j["rs_ma"] = j["rs"].rolling(rs_window).mean()
    j["price_ma"] = j["Stock"].rolling(sma).mean()

    signal = (
        (j["rs"] > j["rs_ma"]) &
        (j["Stock"] > j["price_ma"])
    ).astype(int)

    j["position"] = signal.shift(1).fillna(0)
    j["turnover"] = (
        j["position"].diff().abs().fillna(j["position"].abs())
    )

    j["strategy_ret"] = (
        j["position"] * j["stock_ret"]
        - j["turnover"] * cost_bps / 10000
    )

    j["equity"] = capital * (1 + j["strategy_ret"]).cumprod()
    j["buy_hold"] = capital * (1 + j["stock_ret"]).cumprod()
    j["drawdown"] = j["equity"] / j["equity"].cummax() - 1

    daily = j["strategy_ret"]
    std = daily.std()

    metrics = {
        "Strategy return %": (j["equity"].iloc[-1] / capital - 1) * 100,
        "Buy & hold return %": (j["buy_hold"].iloc[-1] / capital - 1) * 100,
        "Max drawdown %": j["drawdown"].min() * 100,
        "Annualized volatility %": std * np.sqrt(252) * 100,
        "Sharpe (risk-free 0)": (
            daily.mean() / std * np.sqrt(252) if std > 0 else np.nan
        ),
        "Position changes": int((j["turnover"] > 0).sum()),
        "Observations": len(j)
    }

    return j, metrics


# --------------------------------------------------
# 5. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.header("Data settings")

    source = st.radio("Data source", ["Yahoo Finance", "CSV upload"])
    exchange = st.selectbox("Exchange", ["NSE (.NS)", "BSE (.BO)"])

    start = st.date_input(
        "Start date",
        value=date.today() - timedelta(days=730),
        max_value=date.today()
    )

    end = st.date_input(
        "End date (exclusive)",
        value=date.today() + timedelta(days=1),
        min_value=start + timedelta(days=1)
    )

    benchmark_default = "^NSEI" if exchange.startswith("NSE") else "^BSESN"
    benchmark = st.text_input("Benchmark ticker", benchmark_default)

    stock_csv = None
    bench_csv = None
    csv_symbol = None

    if source == "CSV upload":
        stock_csv = st.file_uploader("Stock OHLCV CSV", type=["csv"])
        bench_csv = st.file_uploader(
            "Benchmark OHLCV CSV", type=["csv"], key="benchmark_csv"
        )

        if stock_csv:
            try:
                stock_csv.seek(0)
                temp = pd.read_csv(stock_csv)
                if "Symbol" in temp.columns:
                    symbols = sorted(
                        temp["Symbol"].dropna().astype(str).unique()
                    )
                    csv_symbol = st.selectbox("CSV Symbol", symbols)
            except Exception as exc:
                st.error(f"CSV পড়া যায়নি: {exc}")

    st.caption("Yahoo Finance ডেটা exchange-certified live feed নয়.")


# --------------------------------------------------
# 6. TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard", "Relative Strength", "Scanner",
    "Backtesting", "Data Quality"
])


# DASHBOARD
with tab1:
    st.subheader("Dashboard")
    symbol = st.text_input("Stock ticker", "RELIANCE.NS", key="dashboard_symbol")

    if st.button("Load dashboard"):
        try:
            stock, bench = load_pair(
                symbol, source, stock_csv, bench_csv,
                csv_symbol, start, end, benchmark
            )

            j = joined_data(stock, bench)

            if j.empty:
                st.error("Stock ও benchmark-এর common trading dates নেই।")
            else:
                sr = ret(j["Stock"], 20)
                br = ret(j["Benchmark"], 20)

                c1, c2, c3 = st.columns(3)
                c1.metric("Latest close", f"₹{j['Stock'].iloc[-1]:,.2f}")
                c2.metric(
                    "Stock 20D return",
                    f"{sr:.2f}%" if pd.notna(sr) else "N/A"
                )
                c3.metric(
                    "20D excess return",
                    f"{sr-br:.2f} pp"
                    if pd.notna(sr) and pd.notna(br) else "N/A"
                )

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=j.index, y=j["Stock"], name="Stock close"
                ))
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(j.tail(10), use_container_width=True)

        except Exception as exc:
            st.error(f"Dashboard error: {exc}")


# RELATIVE STRENGTH
with tab2:
    st.subheader("Relative Strength")
    symbol = st.text_input("Stock ticker", "HINDUNILVR.NS", key="rs_symbol")

    periods = st.multiselect(
        "Return windows (trading days)",
        [5, 10, 20, 60, 120, 252],
        default=[20, 60, 120]
    )

    if st.button("Calculate RS"):
        try:
            stock, bench = load_pair(
                symbol, source, stock_csv, bench_csv,
                csv_symbol, start, end, benchmark
            )

            j = joined_data(stock, bench)

            if len(j) < 2:
                st.error("Common trading data অপর্যাপ্ত।")
            else:
                st.dataframe(
                    rs_stats(j, periods or [20]).round(4),
                    use_container_width=True
                )

                ratio = j["Stock"] / j["Benchmark"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=j.index, y=j["Stock"], name="Stock close"
                ))
                fig.add_trace(go.Scatter(
                    x=j.index, y=j["Benchmark"], name="Benchmark close"
                ))
                fig.add_trace(go.Scatter(
                    x=j.index,
                    y=ratio / ratio.iloc[0] * 100,
                    name="Normalized RS",
                    yaxis="y2"
                ))
                fig.update_layout(
                    template="plotly_dark",
                    height=470,
                    yaxis2=dict(overlaying="y", side="right")
                )
                st.plotly_chart(fig, use_container_width=True)

                st.info(
                    "Excess Return = Stock Return − Benchmark Return. "
                    "এটি RSI নয় এবং একা buy/sell signal নয়।"
                )

        except Exception as exc:
            st.error(f"RS error: {exc}")


# SCANNER
with tab3:
    st.subheader("Manual-universe scanner")
    st.caption("এটি বেছে নেওয়া ticker-গুলো স্ক্যান করে; সম্পূর্ণ NSE/BSE universe নয়।")

    raw = st.text_area(
        "Symbols comma-separated",
        "RELIANCE,HINDUNILVR,ICICIBANK,INFY,TCS"
    )

    c1, c2, c3 = st.columns(3)
    trend_on = c1.checkbox("Close > SMA50", True)
    rs_on = c2.checkbox("20D excess return threshold", True)
    vol_on = c3.checkbox("Volume > 20D average", False)

    threshold = st.number_input("Minimum excess return (pp)", value=0.0, step=0.5)
    limit = st.slider("Maximum symbols", 1, 30, 10)

    if st.button("Run scanner"):
        if source != "Yahoo Finance":
            st.warning("Multi-symbol scanner-এর জন্য Yahoo Finance নির্বাচন করুন।")
        else:
            symbols = [
                x.strip().upper()
                for x in raw.replace("\n", ",").split(",")
                if x.strip()
            ][:limit]

            suffix = ".NS" if exchange.startswith("NSE") else ".BO"
            symbols = [
                x if "." in x or x.startswith("^") else x + suffix
                for x in symbols
            ]

            rows, errors = [], []

            try:
                bench = yahoo(benchmark, start, end)
            except Exception as exc:
                bench = None
                st.error(f"Benchmark error: {exc}")

            if bench is not None:
                progress = st.progress(0)

                for i, ticker in enumerate(symbols):
                    try:
                        stock = yahoo(ticker, start, end)
                        j = joined_data(stock, bench)
                        ind = indicators(stock)

                        if len(j) < 55:
                            raise ValueError("কমপক্ষে প্রায় 55 common observations দরকার।")

                        sr = ret(j["Stock"], 20)
                        br = ret(j["Benchmark"], 20)
                        excess = sr - br if pd.notna(sr) and pd.notna(br) else np.nan

                        last = ind.iloc[-1]
                        ok_trend = bool(
                            pd.notna(last["SMA50"])
                            and last["Close"] > last["SMA50"]
                        )
                        ok_vol = bool(
                            pd.notna(last["AvgVol20"])
                            and last["Volume"] > last["AvgVol20"]
                        )

                        checks = []
                        if trend_on:
                            checks.append(ok_trend)
                        if rs_on:
                            checks.append(
                                pd.notna(excess) and excess > threshold
                            )
                        if vol_on:
                            checks.append(ok_vol)

                        rows.append({
                            "Ticker": ticker,
                            "Close": last["Close"],
                            "Stock 20D %": sr,
                            "Benchmark 20D %": br,
                            "Excess pp": excess,
                            "Close>SMA50": ok_trend,
                            "Volume>Avg20": ok_vol,
                            "PASS": all(checks) if checks else True
                        })

                    except Exception as exc:
                        errors.append({"Ticker": ticker, "Error": str(exc)})

                    progress.progress((i + 1) / max(1, len(symbols)))

                if rows:
                    result = pd.DataFrame(rows).sort_values(
                        ["PASS", "Excess pp"],
                        ascending=[False, False],
                        na_position="last"
                    )
                    st.dataframe(result, use_container_width=True)
                    st.download_button(
                        "Download scanner CSV",
                        result.to_csv(index=False).encode("utf-8"),
                        "scanner_results.csv",
                        "text/csv"
                    )
                    st.success(f"Processed {len(rows)} symbols.")

                if errors:
                    st.warning(f"{len(errors)} ticker-এ ডেটা সমস্যা হয়েছে।")
                    st.dataframe(pd.DataFrame(errors), use_container_width=True)

                if not rows:
                    st.warning("কোনো ticker-এর বৈধ ডেটা পাওয়া যায়নি।")


# BACKTEST
with tab4:
    st.subheader("Backtesting")
    st.warning(
        "শিক্ষামূলক baseline. Taxes, slippage, corporate actions, "
        "delisted stocks এবং survivorship bias সম্পূর্ণভাবে model করা হয়নি।"
    )

    symbol = st.text_input("Stock ticker", "HINDUNILVR.NS", key="bt_symbol")

    c1, c2, c3, c4 = st.columns(4)
    rs_window = c1.slider("RS moving average", 5, 100, 20)
    sma = c2.slider("Price SMA", 20, 200, 50)
    capital = c3.number_input(
        "Initial capital ₹", min_value=1000, value=100000, step=1000
    )
    cost = c4.number_input(
        "Cost per position change (bps)",
        min_value=0, max_value=200, value=10, step=5
    )

    if st.button("Run backtest"):
        try:
            stock, bench = load_pair(
                symbol, source, stock_csv, bench_csv,
                csv_symbol, start, end, benchmark
            )

            bt, metrics = backtest(
                stock, bench, rs_window, sma, capital, cost
            )

            cols = st.columns(4)
            cols[0].metric("Strategy return", f"{metrics['Strategy return %']:.2f}%")
            cols[1].metric("Buy & hold", f"{metrics['Buy & hold return %']:.2f}%")
            cols[2].metric("Max drawdown", f"{metrics['Max drawdown %']:.2f}%")
            cols[3].metric("Position changes", str(metrics["Position changes"]))

            st.write(metrics)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=bt.index, y=bt["equity"], name="Strategy equity"
            ))
            fig.add_trace(go.Scatter(
                x=bt.index, y=bt["buy_hold"], name="Buy & hold"
            ))
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

            dd = go.Figure(go.Scatter(
          
