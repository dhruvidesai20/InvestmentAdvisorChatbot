import streamlit as st
import spacy
import yfinance as yf
from transformers import pipeline
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from spacy.matcher import Matcher
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf, adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from pmdarima import auto_arima

nlp = spacy.load("en_core_web_sm")


high_volatility = []
profitable= []

def add_patterns(matcher):
    patterns= [
        {"label": "INVESTMENT_GOAL", "pattern": [{"LOWER": "retirement"}]},
        {"label": "INVESTMENT_GOAL", "pattern": [{"LOWER": "house"}]},
        {"label": "INVESTMENT_GOAL", "pattern": [{"LOWER": "education"}]},
        {"label": "RISK_TOLERANCE", "pattern": [{"LOWER": "high"}, {"LOWER": "risk"}]},
        {"label": "RISK_TOLERANCE", "pattern": [{"LOWER": "low"}, {"LOWER": "risk"}]},
        {"label": "RISK_TOLERANCE", "pattern": [{"LOWER": "medium"}, {"LOWER": "risk"}]},
        {"label": "INVESTMENT_HORIZON", "pattern": [{"IS_DIGIT": True}, {"LOWER": "year"}]},
        {"label": "PREFERRED_SECTORS", "pattern": [{"LOWER": "technology"}]},
        {"label": "PREFERRED_SECTORS", "pattern": [{"LOWER": "healthcare"}]},
        {"label": "PREFERRED_SECTORS", "pattern": [{"LOWER": "finance"}]},
        {"label": "VOLATILITY_TOLERANCE", "pattern": [{"LOWER": "low"}, {"LOWER": "volatility"}]},
        {"label": "VOLATILITY_TOLERANCE", "pattern": [{"LOWER": "medium"}, {"LOWER": "volatility"}]},
    ]
    for pattern in patterns:
        matcher.add(pattern["label"], [pattern["pattern"]])

def preprocess(text):
    doc= nlp(text)
    no_stop_words = [token.text for token in doc if not token.is_stop and not token.is_punct]
    print(no_stop_words)

    filtered_tokens= []

    for token in doc:
        if not token.is_stop and not token.is_punct:
            filtered_tokens.append(token.lemma_)
        else:
            continue
    filtered_text= " ".join(filtered_tokens)
    print(filtered_text)

    doc= nlp(filtered_text)
    matcher= Matcher(nlp.vocab)
    add_patterns(matcher)
    matches = matcher(doc)

    info = {
        "INVESTMENT_GOAL": ["family"],
        "RISK_TOLERANCE": ["medium risk"],
        "INVESTMENT_AMOUNT": ["$10,000"],
        "INVESTMENT_HORIZON": ["1 year"],
        "VOLATILITY_TOLERANCE": ["medium volatility"],
        "PREFERRED_SECTORS": ["technology"]
    }
    
    for match_id, start, end in matches:
        label = nlp.vocab.strings[match_id]
        info[label] = [doc[start:end].text]

    for ent in doc.ents:
        print(ent.text, " | ", ent.label_, " | ", spacy.explain(ent.label_))
        if ent.label_ == "MONEY":
            info["INVESTMENT_AMOUNT"]= [ent.text]

    return info

def fetch_stocks(preferred_sector= None):
    sector_stocks = {
        "technology": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "IBM", "INTC", "ORCL"],
        "finance": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "BRK-B", "MA", "V"],
        "healthcare": ["PFE", "JNJ", "MRK", "ABBV", "AMGN", "GILD", "BMY", "CVS", "UNH", "LLY"],
    }

    if preferred_sector:
        tickers = []
        for sector in preferred_sector:
            if sector in sector_stocks:
                tickers.extend(sector_stocks[sector])
        tickers = list(set(tickers))
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "IBM", "INTC", "ORCL"]

    stock_data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            stock_data[ticker] = {
                "Company": info.get("shortName", ticker),
                "Sector": info.get("sector", "N/A"),
                "Industry": info.get("industry", "N/A"),
                "Market Cap": info.get("marketCap", "N/A"),
                "Current Price": info.get("regularMarketPrice", "N/A")
            }
        except Exception as e:
            print(f"Could not fetch data for {ticker}: {e}")

    sorted_stocks = sorted(stock_data.items(), key=lambda x: x[1].get("Market Cap", 0), reverse=True)[:10]
    top_stocks_df = pd.DataFrame([stock[1] for stock in sorted_stocks], index=[stock[0] for stock in sorted_stocks])

    return top_stocks_df

def plot_stocks(stockData):
    stockData['Close'].resample('M').mean().plot(kind='bar', figsize=(10,6), color='#FFB3C1')
    plt.savefig('monthly_average_closing_prices.png')
    stockData['30 days Rolling']= stockData['Close'].rolling(window=30).mean()
    stockData[['Close', '30 days Rolling']].plot(figsize=(10,6))
    plt.savefig('rolling_average_closing_prices.png')
    stockData['SMA']= stockData['Close'].rolling(window=10, min_periods=1).mean()
    stockData[['Close', 'SMA']].plot(figsize=(10,6))
    plt.savefig('simple_moving_average_closing_prices.png')
    stockData['CMA']= stockData['Close'].expanding().mean()
    stockData[['Close', 'CMA']].plot(figsize=(10,6))
    plt.savefig('cumulative_moving_average_closing_prices.png')
    stockData['EMA']= stockData['Close'].ewm(span=10, adjust=False).mean()
    stockData[['Close', 'EMA']].plot(figsize=(10,6))
    plt.savefig('exponential_moving_average_closing_prices.png')

def predict_stocks(ticker, investment_horizon, text):
    start_date= "2018-1-1"
    end_date= "2024-1-1"

    tickerData = yf.Ticker(ticker)
    stockData = tickerData.history(period='1d',start=start_date, end=end_date)
    stockData= stockData[['Close']]
    stockData.index = stockData.index.date
    stockData.index = pd.to_datetime(stockData.index)

    #Checking for null values
    print("Missing values: ", stockData.isnull().sum())

    stockData.info()
    stockData.describe()

    plot_stocks(stockData)

    plt.figure(figsize=(10,6))
    plt.plot(stockData.Close, color='#00C9BB')
    plt.title('Stock Price over Time (%s)'%ticker, fontsize=20, color='#FE5E54')
    plt.xlabel('Date', fontsize=16)
    plt.ylabel('Close Price', fontsize=16)
    plt.axhline(stockData.Close.mean(), color='#E2E0E5', linestyle='--')

    plt.savefig('stock_data(%s).png'%ticker)

    stepwise_fit= auto_arima(stockData.Close, seasonal=True, m=12, trace=True)
    p, d, q = stepwise_fit.order

    print(f"p: {p}, d: {d}, q: {q}")

    model= SARIMAX(stockData.Close, order= (p,d,q), seasonal_order=(0,d,0,12))
    model= model.fit()

    forecast_steps = int(investment_horizon[0].split()[0])*252
    forecast = model.forecast(steps=forecast_steps)

    future_dates = pd.date_range(start=stockData.index[-1], periods=forecast_steps)
    forecast.index= future_dates
    volatility_calc(stockData, text, ticker)

    exp_return= ((forecast[-1]- forecast[0])/forecast[0])*100
    if exp_return>0:
        profitable.append(ticker)
    return forecast

def volatility_calc(stockData, text, ticker):
    stockData['Daily Return'] = stockData['Close'].pct_change()

    volatility = stockData['Daily Return'].std() * 100
    extracted_info= preprocess(text)
    volatility_user= extracted_info.get("VOLATILITY_TOLERANCE")[0]
    analyze_stock(stockData, volatility, volatility_user, ticker)

def analyze_stock(stockData, volatility, volatility_user, ticker):
    volatility_thresholds = {
        'low volatility': 10.0,
        'medium volatility': 20.0,
        'high volatility': float('inf')
    }

    max_volatility = volatility_thresholds[volatility_user]

    if not (volatility <= max_volatility):
            st.write(f"{ticker} has volatility of {volatility:.2f}%, which is outside the {volatility_user} range.")
            high_volatility.append(ticker)

def main():
    st.title("Investment Advisor")

    user_input = st.text_area("Enter your investment preferences:")

    if st.button("Analyze"):
        if user_input:
            extracted_info= preprocess(user_input)
            st.write("Extracted Information:")
            for key, value in extracted_info.items():
                st.write(f"{key}: {', '.join(value)}")
            preferred_sectors = extracted_info.get("PREFERRED_SECTORS")
            top_stocks_df = fetch_stocks(preferred_sectors)
            st.write("Top 10 Stocks:")
            st.dataframe(top_stocks_df)

            investment_horizon = extracted_info.get("INVESTMENT_HORIZON")
            tickers = top_stocks_df.index

            for ticker in tickers:
                st.write(f"Predicting future prices for {ticker}...")
                forecast_df = predict_stocks(ticker, investment_horizon, user_input)

                st.write(f"Predicted Future Prices for {ticker}:")
                plt.figure(figsize=(8, 5))  # Set the size of the plot
                plt.plot(forecast_df, color='#987D9A', linewidth=2)
                st.pyplot(plt)
            
            if len(high_volatility)>0:
                st.write("Comapnies with volatility higher than your tolerance level", high_volatility)

            st.write("List of stocks expected to provide profit: ", profitable)

        else:
            st.write("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
