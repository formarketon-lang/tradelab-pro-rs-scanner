import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import date, timedelta

st.set_page_config(page_title="TradeLab Pro RS", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp{background:#0b1120;color:#e2e8f0}
[data-testid="stSidebar"]{background:#111827}
[data-testid="stMetric"]{background:#1e293b;border:1px solid #334155;padding:12px;border-radius:12px}
</style>""", unsafe_allow_html=True)
st.title("TradeLab Pro — RS + Scanner + Backtesting")
st.caption("NSE/BSE cash-equity research • Historical data • Not a trading execution system")

def normalize(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace("_", " ").strip()
        if k in ("date","datetime","timestamp","time"): rename[c]="Date"
        elif k in ("open","o"): rename[c]="Open"
        elif k in ("high","h"): rename[c]="High"
        elif k in ("low","l"): rename[c]="Low"
        elif k in ("close","c","adj close","adjusted close","adjclose"): 
            if "Close" not in rename.values(): rename[c]="Close"
        elif k in ("volume","vol","v"): rename[c]="Volume"
        elif k in ("symbol","ticker","scrip"): rename[c]="Symbol"
    df = df.rename(columns=rename)
    if "Date" not in df:
        if isinstance(df.index, pd.DatetimeIndex): df = df.reset_index().rename(columns={"index":"Date"})
        else: raise ValueError("CSV-তে Date কলাম আবশ্যক।")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["Date"]).sort_values("Date")
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df:
            if c == "Volume": df[c] = 0
            else: raise ValueError(f"Missing column: {c}")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open","High","Low","Close"])
    df = df[(df["Close"] > 0) & (df["High"] >= df["Low"])]
    df = df.drop_duplicates("Date", keep="last").sort_values("Date").set_index("Date")
    if df.empty: raise ValueError("যাচাইয়ের পরে কোনো valid data নেই।")
    return df[["Open","High","Low","Close","Volume"]].astype(float)

@st.cache_data(ttl=900, show_spinner=False)
def yahoo(ticker, start, end):
    d = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if d is None or d.empty: raise ValueError(f"{ticker}: Yahoo Finance data পাওয়া যায়নি। Ticker/date/network যাচাই করুন।")
    if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
    return normalize(d.reset_index())

def load_pair(symbol, source, stock_csv, bench_csv, csv_symbol, start, end, benchmark):
    if source == "Yahoo Finance":
        s, b = yahoo(symbol, str(start), str(end)), yahoo(benchmark, str(start), str(end))
    else:
        if stock_csv is None or bench_csv is None: raise ValueError("CSV mode-এ Stock এবং Benchmark—দুটি CSV-ই আপলোড করুন।")
        sf, bf = pd.read_csv(stock_csv), pd.read_csv(bench_csv)
        if "Symbol" in sf.columns:
            sf = sf[sf["Symbol"].astype(str).str.upper() == str(csv_symbol).upper()].drop(columns=["Symbol"])
        s, b = normalize(sf), normalize(bf)
    return s.loc[str(start):str(end)], b.loc[str(start):str(end)]

def ret(series, n):
    return (series.iloc[-1] / series.iloc[-n-1] - 1)*100 if len(series) > n else np.nan

def joined_data(stock, bench):
    return pd.concat([stock["Close"].rename("Stock"), bench["Close"].rename("Benchmark")], axis=1).dropna()

def rs_stats(j, periods):
    out=[]
    for n in periods:
        sr, br = ret(j.Stock,n), ret(j.Benchmark,n)
        out.append({"Window (trading days)":n,"Stock Return %":sr,"Benchmark Return %":br,
                    "Excess Return (pp)":sr-br if pd.notna(sr) and pd.notna(br) else np.nan,
                    "Relative Ratio":(1+sr/100)/(1+br/100) if pd.notna(sr) and pd.notna(br) and br != -100 else np.nan,
                    "Common dates":len(j)})
    return pd.DataFrame(out)

def indicators(s):
    x=s.copy()
    x["SMA50"]=x.Close.rolling(50).mean()
    x["AvgVol20"]=x.Volume.rolling(20).mean()
    x["High20Prev"]=x.High.shift(1).rolling(20).max()
    return x

def backtest(s, b, rs_window, sma, capital, cost_bps):
    j=joined_data(s,b).copy()
    j["stock_ret"]=j.Stock.pct_change().fillna(0)
    j["rs"]=j.Stock/j.Benchmark
    j["rs_ma"]=j.rs.rolling(rs_window).mean()
    j["price_ma"]=j.Stock.rolling(sma).mean()
    signal=((j.rs>j.rs_ma)&(j.Stock>j.price_ma)).astype(int)
    j["position"]=signal.shift(1).fillna(0)  # signal from prior close, no same-bar look-ahead
    j["turnover"]=j.position.diff().abs().fillna(j.position.abs())
    j["strategy_ret"]=j.position*j.stock_ret-j.turnover*(cost_bps/10000)
    j["equity"]=capital*(1+j.strategy_ret).cumprod()
    j["buy_hold"]=capital*(1+j.stock_ret).cumprod()
    j["drawdown"]=j.equity/j.equity.cummax()-1
    daily=j.strategy_ret
    metrics={"Strategy return %":(j.equity.iloc[-1]/capital-1)*100,
             "Buy & hold return %":(j.buy_hold.iloc[-1]/capital-1)*100,
             "Max drawdown %":j.drawdown.min()*100,
             "Annualized volatility %":daily.std()*np.sqrt(252)*100,
             "Sharpe (risk-free 0)":daily.mean()/daily.std()*np.sqrt(252) if daily.std()>0 else np.nan,
             "Position changes":int((j.turnover>0).sum()),"Observations":len(j)}
    return j,metrics

with st.sidebar:
    st.header("Data settings")
    source=st.radio("Data source",["Yahoo Finance","CSV upload"])
    exchange=st.selectbox("Plain ticker suffix",["NSE (.NS)","BSE (.BO)"])
    start=st.date_input("Start date",date.today()-timedelta(days=730),max_value=date.today())
    end=st.date_input("End date (exclusive)",date.today()+timedelta(days=1),min_value=start+timedelta(days=1))
    benchmark=st.text_input("Benchmark ticker","^NSEI" if exchange.startswith("NSE") else "^BSESN")
    stock_csv=bench_csv=None
    csv_symbol=None
    if source=="CSV upload":
        stock_csv=st.file_uploader("Stock OHLCV CSV",type=["csv"])
        bench_csv=st.file_uploader("Benchmark OHLCV CSV",type=["csv"],key="benchmark")
        if stock_csv:
            try:
                temp=pd.read_csv(stock_csv)
                if "Symbol" in temp.columns:
                    opts=sorted(temp.Symbol.dropna().astype(str).unique())
                    csv_symbol=st.selectbox("Symbol in stock CSV",opts)
            except Exception as e: st.error(str(e))
    st.caption("Yahoo/yfinance হলো convenience research source; guaranteed live বা exchange-certified feed নয়।")

tab1,tab2,tab3,tab4,tab5=st.tabs(["Dashboard","Relative Strength","Scanner","Backtesting","Data Quality"])

with tab1:
    st.subheader("Dashboard")
    sym=st.text_input("Stock ticker", "RELIANCE.NS",key="dash")
    if st.button("Load dashboard"):
        try:
            s,b=load_pair(sym,source,stock_csv,bench_csv,csv_symbol,start,end,benchmark)
            j=joined_data(s,b)
            if j.empty: st.error("Stock ও benchmark-এর common dates নেই।")
            else:
                sr,br=ret(j.Stock,20),ret(j.Benchmark,20)
                a,c,d=st.columns(3)
                a.metric("Latest close",f"₹{j.Stock.iloc[-1]:,.2f}")
                c.metric("Stock 20D return",f"{sr:.2f}%" if pd.notna(sr) else "N/A")
                d.metric("20D excess return",f"{sr-br:.2f} pp" if pd.notna(sr) and pd.notna(br) else "N/A")
                fig=go.Figure(); fig.add_trace(go.Scatter(x=j.index,y=j.Stock,name="Stock"))
                fig.update_layout(template="plotly_dark",height=400); st.plotly_chart(fig,use_container_width=True)
                st.dataframe(j.tail(10),use_container_width=True)
        except Exception as e: st.error(str(e))

with tab2:
    st.subheader("Relative Strength")
    sym=st.text_input("Stock ticker", "HINDUNILVR.NS",key="rs")
    periods=st.multiselect("Return windows (trading days)",[5,10,20,60,120,252],[20,60,120])
    if st.button("Calculate RS"):
        try:
            s,b=load_pair(sym,source,stock_csv,bench_csv,csv_symbol,start,end,benchmark)
            j=joined_data(s,b)
            if len(j)<2: st.error("কমন ট্রেডিং ডেটা অপর্যাপ্ত।")
            else:
                st.dataframe(rs_stats(j,periods or [20]).round(4),use_container_width=True)
                ratio=j.Stock/j.Benchmark
                fig=go.Figure()
                fig.add_trace(go.Scatter(x=j.index,y=j.Stock,name="Stock close"))
                fig.add_trace(go.Scatter(x=j.index,y=j.Benchmark,name="Benchmark close"))
                fig.add_trace(go.Scatter(x=j.index,y=ratio/ratio.iloc[0]*100,name="Normalized RS (start=100)",yaxis="y2"))
                fig.update_layout(template="plotly_dark",height=470,yaxis2=dict(overlaying="y",side="right",title="Normalized RS"))
                st.plotly_chart(fig,use_container_width=True)
                st.info("Excess Return = Stock Return − Benchmark Return (percentage points). Relative Ratio = (1 + stock return)/(1 + benchmark return). এটি RSI নয় এবং একা buy/sell signal নয়।")
        except Exception as e: st.error(str(e))

with tab3:
    st.subheader("Manual-universe scanner")
    st.caption("প্রথম সংস্করণে Yahoo Finance data source-এ scanner চলে। Verified full NSE/BSE symbol master পরে যুক্ত করতে হবে।")
    raw=st.text_area("Symbols comma-separated", "RELIANCE,HINDUNILVR,ICICIBANK,INFY,TCS")
    x,y,z=st.columns(3)
    trend=x.checkbox("Close > SMA50",True)
    rs_on=y.checkbox("20D excess return > threshold",True)
    vol_on=z.checkbox("Volume > 20D average",False)
    threshold=st.number_input("Minimum excess return (pp)",0.0,step=0.5)
    limit=st.slider("Maximum symbols",1,30,10)
    if st.button("Run scanner"):
        if source!="Yahoo Finance": st.warning("Scanner multi-symbol mode আপাতত Yahoo Finance-এর জন্য; CSV mode-এ RS ও backtest চলে।")
        else:
            syms=[q.strip().upper() for q in raw.replace("\\n",",").split(",") if q.strip()][:limit]
            if exchange.startswith("NSE"): syms=[q if "." in q or q.startswith("^") else q+".NS" for q in syms]
            else: syms=[q if "." in q or q.startswith("^") else q+".BO" for q in syms]
            try: bench=yahoo(benchmark,str(start),str(end))
            except Exception as e: st.error(f"Benchmark error: {e}"); bench=None
            rows=[]; errors=[]
            if bench is not None:
                progress=st.progress(0)
                for i,t in enumerate(syms):
                    try:
                        s=yahoo(t,str(start),str(end)); j=joined_data(s,bench); ind=indicators(s)
                        if len(j)<55: raise ValueError("প্রায় 55+ common trading observations দরকার।")
                        sr,br=ret(j.Stock,20),ret(j.Benchmark,20); excess=sr-br
                        last=ind.iloc[-1]
                        ok_trend=bool(pd.notna(last.SMA50) and last.Close>last.SMA50)
                        ok_vol=bool(pd.notna(last.AvgVol20) and last.Volume>last.AvgVol20)
                        checks=[]
                        if trend: checks.append(ok_trend)
                        if rs_on: checks.append(bool(pd.notna(excess) and excess>threshold))
                        if vol_on: checks.append(ok_vol)
                        rows.append({"Ticker":t,"Close":last.Close,"Stock 20D %":sr,"Benchmark 20D %":br,"Excess pp":excess,
                                     "Close>SMA50":ok_trend,"Volume>Avg20":ok_vol,"PASS":all(checks) if checks else True})
                    except Exception as e: errors.append({"Ticker":t,"Error":str(e)})
                    progress.progress((i+1)/max(1,len(syms)))
                if rows:
                    result=pd.DataFrame(rows).sort_values(["PASS","Excess pp"],ascending=[False,False])
                    st.dataframe(result,use_container_width=True)
                    st.download_button("Download scanner CSV",result.to_csv(index=False).encode(), "scanner_results.csv","text/csv")
                if errors: st.warning("কিছু ticker retrieve/validate করা যায়নি।"); st.dataframe(pd.DataFrame(errors),use_container_width=True)

with tab4:
    st.subheader("Backtesting")
    st.warning("Educational baseline. Taxes, realistic slippage, corporate actions, delisted stocks and survivorship bias are not fully modelled.")
    sym=st.text_input("Stock ticker","HINDUNILVR.NS",key="bt")
    c1,c2,c3,c4=st.columns(4)
    rs_window=c1.slider("RS moving average",5,100,20)
    sma=c2.slider("Price SMA",20,200,50)
    capital=c3.number_input("Initial capital ₹",min_value=1000,value=100000,step=1000)
    cost=c4.number_input("Cost per position change (bps)",min_value=0,max_value=200,value=10,step=5)
    if st.button("Run backtest"):
        try:
            s,b=load_pair(sym,source,stock_csv,bench_csv,csv_symbol,start,end,benchmark)
            bt,m=backtest(s,b,rs_window,sma,capital,cost)
            cols=st.columns(4)
            cols[0].metric("Strategy return",f"{m['Strategy return %']:.2f}%")
            cols[1].metric("Buy & hold",f"{m['Buy & hold return %']:.2f}%")
            cols[2].metric("Max drawdown",f"{m['Max drawdown %']:.2f}%")
            cols[3].metric("Position changes",str(m["Position changes"]))
            st.write({k:round(v,4) if isinstance(v,(float,np.floating)) and pd.notna(v) else v for k,v in m.items()})
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=bt.index,y=bt.equity,name="Strategy equity"))
            fig.add_trace(go.Scatter(x=bt.index,y=bt.buy_hold,name="Buy & hold"))
            fig.update_layout(template="plotly_dark",height=400); st.plotly_chart(fig,use_container_width=True)
            dd=go.Figure(go.Scatter(x=bt.index,y=bt.drawdown*100,name="Drawdown"))
            dd.update_layout(template="plotly_dark",height=250,title="Drawdown (%)"); st.plotly_chart(dd,use_container_width=True)
            st.download_button("Download backtest CSV",bt.reset_index().to_csv(index=False).encode(),"backtest.csv","text/csv")
        except Exception as e: st.error(str(e))

with tab5:
    st.subheader("Data quality")
    st.markdown("""
    **Validation performed**
    - Date parse, chronological sorting, duplicate-date removal.
    - Numeric OHLCV checks; positive Close; High ≥ Low.
    - Stock/benchmark calculations use only common dates.
    - Return windows count trading observations, not calendar days.
    - Backtest positions are shifted one bar to reduce look-ahead bias.

    **Stock CSV:** `Date,Symbol,Open,High,Low,Close,Volume`  
    **Benchmark CSV:** `Date,Open,High,Low,Close,Volume`

    **Important limitations:** Yahoo data may be delayed, adjusted or missing. For production use, add a licensed broker/exchange feed, verified symbol master, corporate-action controls, raw-data audit trail and survivorship-bias controls.
    """)
    st.code("Date,Symbol,Open,High,Low,Close,Volume\n2026-01-02,RELIANCE,1500,1520,1490,1510,1234567",language="csv")

st.divider()
st.caption("Research tool only. Verify prices and results independently; no live order placement.")
                                        
