import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objs as go

#Main function

st.sidebar.title("Navigation")
app_selection = st.sidebar.radio("Go to", ["Stock Dashboard", "Stock Prediction"])

# Stock Dashboard App
if app_selection == "Stock Dashboard":
    def load_data(stock,period):
        df=yf.Ticker(stock).history(period=period)[['Open','High','Low','Close','Volume']]
        #data wrangling
        df=df.reset_index()
        df.columns=['time','open','high','low','close','volume']
        df['time']=df['time'].dt.strftime("%Y-%m-%d")
        df.ta.ema(length=20,append=True)
        df.ta.ema(length=200,append=True)
        df.ta.rsi(length=14,append=True)
        df.ta.adx(length=14,append=True)
        df.ta.atr(length=14,append=True)
        return df

    #get emojis
    def get_returns_emoji(ret_val):
        emoji=":white_check_mark:"
        if ret_val<0:
            emoji=":red_circle:"
        return emoji
    #ema emoji
    def get_ema_emoji(ltp,ema):
        emoji=":white_check_mark:"
        if ltp<ema:
            emoji=":red_circle:"
        return emoji    
    #rsi emoji
    def get_rsi_emoji(rsi):
        emoji=":red_circle:"
        if rsi>30 and rsi<70:
            emoji=":white_check_mark:"
        return emoji
    def get_adx_emoji(adx):
        emoji=":red_circle:"
        if adx>25:
            emoji=":white_check_mark:"
        return emoji

    #Chart creation
    def create_chart(df):
        candlestick_chart=go.Figure(data=[go.Candlestick(x=df.index,open=df['open'],high=df['high'],low=df['low'],close=df['close'])])
        ema20=go.Scatter(x=df.EMA_20.index,y=df.EMA_20.values,name='EMA20')
        ema200=go.Scatter(x=df.EMA_200.index,y=df.EMA_200.values,name='EMA200')
        candlestick_chart.update_layout(title=f'{stock}Historical Candlestick Chart',
                                        xaxis_title='Date',
                                        yaxis_title='Price',
                                        xaxis_rangeslider_visible=True)
        candlestick_chart.add_trace(ema20)
        candlestick_chart.add_trace(ema200)
        return candlestick_chart
            
    st.title(":rainbow[Stock Analysis Dashboard]")
    #Sidebar
    stock=st.sidebar.text_input("Stock Symbol ex: AAPL","AAPL")     
    timeframe_option=st.sidebar.selectbox("Timeframe?",('1y','1d', '5d', '1mo', '3mo','6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'))
    show_data=st.sidebar.checkbox(label="Show Data")
    show_chart=st.sidebar.checkbox(label="Show Chart")

    df=load_data(stock,timeframe_option)
    reversed_df=df.iloc[::-1]
    row1_val=reversed_df.iloc[0]['close']
    ema20_val=reversed_df.iloc[0]['EMA_20']
    ema200_val=reversed_df.iloc[0]['EMA_200']
    rsi_val=reversed_df.iloc[0]['RSI_14']
    adx=reversed_df.iloc[0]['ADX_14']
    dmp=reversed_df.iloc[0]['DMP_14']
    dmn=reversed_df.iloc[0]['DMN_14']
    row20_val=reversed_df.iloc[20]['close']#1 month return
    row60_val=reversed_df.iloc[60]['close']#3 months return
    row120_val=reversed_df.iloc[120]['close']#6 month return
    row240_val=reversed_df.iloc[240]['close']#12 months return

    #Return % Calculation
    day20_ret_percent=(row1_val-row20_val)/row20_val*100
    day20_ret_val=(row1_val-row20_val)
    day60_ret_percent=(row1_val-row60_val)/row60_val*100
    day60_ret_val=(row1_val-row60_val)
    day120_ret_percent=(row1_val-row120_val)/row120_val*100
    day120_ret_val=(row1_val-row120_val)
    day240_ret_percent=(row1_val-row240_val)/row240_val*100
    day240_ret_val=(row1_val-row240_val)

    #Column display
    col1,col2,col3=st.columns(3)
    with col1:
        st.subheader("Returns")
        st.markdown(f"-1 MONTH : {round(day20_ret_percent,2)}%{get_returns_emoji(round(day20_ret_percent,2))}")
        st.markdown(f"-3 MONTH : {round(day60_ret_percent,2)}%{get_returns_emoji(round(day60_ret_percent,2))}")
        st.markdown(f"-6 MONTH : {round(day120_ret_percent,2)}%{get_returns_emoji(round(day120_ret_percent,2))}")
        st.markdown(f"-12 MONTH : {round(day240_ret_percent,2)}%{get_returns_emoji(round(day240_ret_percent,2))}")
    with col2:
        st.subheader("Momentum")
        st.markdown(f"- LTP : {round(row1_val,2)}") 
        st.markdown(f"- EMA20 :{round(ema20_val,2)}{get_ema_emoji(round(row1_val,2),round(ema20_val,2))}")
        st.markdown(f"- EMA200 :{round(ema200_val,2)}{get_ema_emoji(round(row1_val,2),round(ema200_val,2))}")   
        st.markdown(f"- RSI :{round(rsi_val,2)}{get_rsi_emoji(round(rsi_val,2))}")
    with col3:
        st.subheader("Trend Strength")
        st.markdown(f"- ADX : {round(adx,2)}{get_adx_emoji(round(adx,2))}")
        st.markdown(f"- DMP : {round(dmp,2)}")
        st.markdown(f"- DMN : {round(dmn,2)}")

    if show_data:
        st.write(reversed_df)

    if show_chart:
        st.plotly_chart(create_chart(df))
elif app_selection == "Stock Prediction":
    import streamlit as st
    from datetime import date
    import yfinance as yf
    from prophet import Prophet
    from prophet.plot import plot_plotly
    from plotly import graph_objs as go

    START="2015-01-01"
    TODAY=date.today().strftime("%Y-%m-%d")

    st.title(":rainbow[Stock Prediction App using Machine Learning]")

    stocks=("AAPL","GOOG","MSFT","GME")
    selected_stocks=st.selectbox("Select dataset for prediction",stocks)
    n_years=st.slider("Years of Predictions:",1,4)
    period=n_years*365

    @st.cache_data
    def load_data(ticker):
        data=yf.download(ticker,START,TODAY)
        data.reset_index(inplace=True)
        return data
    data_load_state=st.text("Load data...")
    data=load_data(selected_stocks)
    data_load_state.text("Loading data Done...")

    st.subheader('Raw Data')
    st.write(data.tail())

    def plot_raw_data():
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Open'],name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'],y=data['Close'],name='stock_close'))
        fig.layout.update(title_text="Time Series Data",xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    #Forecasting
    df_train=data[['Date','Close']]
    df_train=df_train.rename(columns={"Date":"ds","Close":"y"})

    m=Prophet()
    m.fit(df_train)
    future=m.make_future_dataframe(periods=period)
    forecast=m.predict(future)

    st.subheader('Forecast Data')
    st.write(forecast.tail())

    st.subheader('Forecast data')
    fig1=plot_plotly(m,forecast)
    st.plotly_chart(fig1)

    st.subheader('Forecast component')
    fig2=m.plot_components(forecast)
    st.write(fig2)