📊 Data-Driven Stock Analysis (Streamlit + MySQL)

A complete end-to-end stock analysis dashboard built using Streamlit, Pandas, MySQL, and Matplotlib.
This tool helps you load raw YAML-based stock data, store it in a MySQL database, export per-stock CSV files, and analyse stocks using multiple analytics modules.

🔥 Features
✅ YAML → DataFrame Loader

Reads multiple monthly YAML files from folders and combines them into a single Pandas DataFrame.

✅ MySQL Integration

Create database (DATA)

Create table (MARKET)

Insert all YAML records into MySQL

Export each stock’s data into individual CSV files

✅ Multi-Page Streamlit Dashboard

Includes the following analytics pages:

Volatility Analysis

Computes daily return volatility

Displays Top 10 Most Volatile Stocks

Cumulative Return Analysis

Calculates cumulative return from the starting point

Plots Top 5 Stocks over time

Sector-wise Performance

Reads sector mapping CSV

Computes yearly return for each stock

Plots average yearly return per sector

Stock Price Correlation

Correlation heatmap of stock closing prices

Top 5 Gainers & Losers (Month-wise)

Computes monthly returns

Shows the best and worst performing stocks per month
