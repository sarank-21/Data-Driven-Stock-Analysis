import os
import glob
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import mysql.connector
import streamlit as st
from datetime import datetime

# -----------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Data-Driven Stock Analysis",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -----------------------------------------------------------
# PATH CONFIG
# -----------------------------------------------------------
folder_path = r"D:\PROJECTS\Project_2\Data-Driven-Stock-Analysis\data"
sector_file = r"D:\PROJECTS\Project_2\Data-Driven-Stock-Analysis\Sector_data - Sheet1.csv"

MONTHS = [
    "2023-10", "2023-11", "2023-12",
    "2024-01", "2024-02", "2024-03",
    "2024-04", "2024-05", "2024-06",
    "2024-07", "2024-08", "2024-09",
    "2024-10", "2024-11"
]

# -----------------------------------------------------------
# MYSQL CONNECTION
# -----------------------------------------------------------
def get_connection(database=None):
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="0007",
            database=database
        )
    except mysql.connector.Error as e:
        st.error(f"MySQL Error: {e}")
        return None

# -----------------------------------------------------------
# DATABASE SETUP
# -----------------------------------------------------------
def setup_database():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS sector")
    cursor.close()
    conn.close()

    conn = get_connection("sector")
    cursor = conn.cursor()

    # MARKET TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MARKET (
            Ticker VARCHAR(50),
            date DATE,
            open DECIMAL(15,6),
            high DECIMAL(15,6),
            low DECIMAL(15,6),
            close DECIMAL(15,6),
            volume BIGINT,
            month VARCHAR(10)
        )
    """)

    # 1.VOLATILITY RESULTS TABLE
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS VOLATILITY_RESULTS (
            Ticker VARCHAR(50),
            Volatility DECIMAL(15,8)
        )
    """)

    # 2.Cumulative_Return  TABLE
    cursor.execute(""" CREATE TABLE IF NOT EXISTS Cumulative_Return (
    Date DATE,
    Close DECIMAL(10,2),
    Ticker VARCHAR(50),
    Start_Close DECIMAL(10,2),
    Cumulative_Return DECIMAL(15,8),
    Cumulative_Return_Percent DECIMAL(15,8)
    )
    """)

    # 3.Sector-wise Performance  TABLE
    cursor.execute(""" CREATE TABLE IF NOT EXISTS Sector_wise (
                   Sector VARCHAR(50),
                   Yearly_Return DECIMAL(15,8))
                   """)

    # 4.Stock Price Correlation TABLE
    cursor.execute(""" CREATE TABLE IF NOT EXISTS Stock_Price_Correlation (
                   stock_1 VARCHAR(20),
                   stock_2 VARCHAR(20),
                   correlation FLOAT)
                   """)


    #  5.Top 5 Gainers  (Month-wise) TABLE
    cursor.execute(""" CREATE TABLE IF NOT EXISTS Top_Gainers (
                   Ticker VARCHAR(50),
                   Month VARCHAR(50),
                   First DECIMAL(15,8),
                   Last DECIMAL(15,8),
                   Monthly_Return DECIMAL(15,8))
                   """)

    conn.commit()
    cursor.close()
    conn.close()
    st.success("✅ Database & tables ready")

# -----------------------------------------------------------
# LOAD YAML
# -----------------------------------------------------------
def load_yaml_data():
    records = []
    for month in MONTHS:
        folder = os.path.join(folder_path, month)
        if not os.path.exists(folder):
            continue
        for file in glob.glob(os.path.join(folder, "*.yaml")):
            with open(file, "r") as f:
                data = yaml.safe_load(f)
                if isinstance(data, list):
                    records.extend(data)
                elif isinstance(data, dict):
                    records.append(data)
    return pd.DataFrame(records)

# -----------------------------------------------------------
# INSERT YAML INTO SQL
# -----------------------------------------------------------
def insert_into_db(df):
    if df.empty:
        st.warning("⚠️ No YAML data found")
        return

    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM MARKET")
    if cursor.fetchone()[0] > 0:
        st.info("ℹ️ YAML data already inserted")
        cursor.close()
        conn.close()
        return

    query = """
        INSERT INTO MARKET
        (Ticker, date, open, high, low, close, volume, month)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """

    for _, row in df.iterrows():
        cursor.execute(query, (
            row.get("Ticker"),
            pd.to_datetime(row.get("date")).date(),
            row.get("open"),
            row.get("high"),
            row.get("low"),
            row.get("close"),
            row.get("volume"),
            row.get("month")
        ))

    conn.commit()
    cursor.close()
    conn.close()
    st.success(f"✅ Inserted {len(df)} rows into MARKET")

# -----------------------------------------------------------
# EXPORT CSV
# -----------------------------------------------------------
def export_csv():
    out_dir = os.path.join(folder_path, "output_csv")
    os.makedirs(out_dir, exist_ok=True)

    conn = get_connection("sector")
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT Ticker FROM MARKET")
    tickers = [t[0] for t in cursor.fetchall()]

    created = 0
    for ticker in tickers:
        path = os.path.join(out_dir, f"{ticker}.csv")
        if os.path.exists(path):
            continue

        cursor.execute("""
            SELECT Ticker, date, open, high, low, close, volume
            FROM MARKET WHERE Ticker=%s ORDER BY date
        """, (ticker,))
        rows = cursor.fetchall()

        df = pd.DataFrame(rows, columns=[
            "Ticker", "date", "open", "high", "low", "close", "volume"
        ])
        df.to_csv(path, index=False)
        created += 1

    cursor.close()
    conn.close()

    if created == 0:
        st.info("ℹ️ CSV files already exist")
    else:
        st.success(f"✅ {created} CSV files created")

# -----------------------------------------------------------
# VOLATILITY + SQL UPLOAD
# -----------------------------------------------------------
def calculate_and_store_volatility():
    output_folder = os.path.join(folder_path, "output_csv")
    all_files = glob.glob(os.path.join(output_folder, "*.csv"))
    if not all_files:
        st.warning("⚠️ No CSV files found")
        return pd.DataFrame()

    list_val = []
    for filename in all_files:
        df = pd.read_csv(filename)
        ticker = os.path.splitext(os.path.basename(filename))[0].upper()

        # ensure required cols exist
        if 'date' not in df.columns or 'close' not in df.columns:
            continue

        df_clean = df[['date', 'close']].copy()
        df_clean.rename(columns={'date': 'Date', 'close': 'Close'}, inplace=True)
        df_clean['Ticker'] = ticker

        list_val.append(df_clean)

    if not list_val:
        return pd.DataFrame(columns=['Ticker','Volatility']), pd.DataFrame()

    combined_df = pd.concat(list_val, ignore_index=True)

    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df['Close'] = pd.to_numeric(combined_df['Close'], errors='coerce')
    combined_df.dropna(subset=['Close'], inplace=True)
    combined_df.sort_values(['Ticker', 'Date'], inplace=True)

    combined_df['Daily_Return'] = combined_df.groupby('Ticker')['Close'].pct_change()
    returns_df = combined_df.dropna(subset=['Daily_Return'])

    volatility = returns_df.groupby('Ticker')['Daily_Return'].std().reset_index()
    volatility.columns = ['Ticker', 'Volatility']

    top10 = volatility.sort_values(by='Volatility', ascending=False).head(10)

    # -------- INSERT INTO SQL --------
    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM VOLATILITY_RESULTS")
    if cursor.fetchone()[0] > 0:
        st.info("ℹ️ Volatility results already stored in database")
    else:
        for _, row in top10.iterrows():
            cursor.execute("""
                INSERT INTO VOLATILITY_RESULTS
                (Ticker, Volatility)
                VALUES (%s,%s)
            """, (row["Ticker"], row["Volatility"]))

        conn.commit()
        st.success("✅ Top 10 Volatile Stocks stored in SQL")

    cursor.close()
    conn.close()

    return top10

# -----------------------------------------------------------
# 2. Cumulative Return Over Time Page
# -----------------------------------------------------------

def cumulative_return_plot(folder_path):
    output_folder = os.path.join(folder_path, "output_csv")
    all_files = glob.glob(os.path.join(output_folder, "*.csv"))

    list_val = []
    DATE_COL = 'date'
    CLOSE_COL = 'close'

    for filename in all_files:
        df2 = pd.read_csv(filename)
        ticker = os.path.splitext(os.path.basename(filename))[0].upper()

        if DATE_COL not in df2.columns or CLOSE_COL not in df2.columns:
            continue

        df_cum = df2[[DATE_COL, CLOSE_COL]].copy()
        df_cum.rename(columns={DATE_COL: 'Date', CLOSE_COL: 'Close'}, inplace=True)
        df_cum['Ticker'] = ticker

        list_val.append(df_cum)

    if not list_val:
        st.warning("No CSV files found in output_csv.")
        return [], pd.DataFrame()

    cum_df = pd.concat(list_val, axis=0, ignore_index=True)

    cum_df['Date'] = pd.to_datetime(cum_df['Date'])
    cum_df['Close'] = pd.to_numeric(cum_df['Close'], errors='coerce')
    cum_df.dropna(subset=['Close'], inplace=True)
    cum_df.sort_values(by=['Ticker', 'Date'], inplace=True)

    # Cumulative return calculation from first available price per ticker
    cum_df['Start_Close'] = cum_df.groupby("Ticker")['Close'].transform('first')
    cum_df['Cumulative_Return'] = (cum_df['Close'] / cum_df['Start_Close']) - 1
    cum_df['Cumulative_Return_Percent'] = cum_df['Cumulative_Return'] * 100

    final_cum = cum_df.groupby("Ticker")['Cumulative_Return'].last().reset_index()
    top5_tickers = final_cum.sort_values(by='Cumulative_Return', ascending=False).head(5)['Ticker'].tolist()

    st.subheader("🏆 Top 5 Stocks Based on Cumulative Return")
    st.write(top5_tickers)

    top5_df = cum_df[cum_df['Ticker'].isin(top5_tickers)]

    st.subheader("📈 Cumulative Return Over Time - Top 5 Performing Stocks")
    fig, ax = plt.subplots(figsize=(14, 7))
    for ticker in top5_tickers:
        df_plot = top5_df[top5_df['Ticker'] == ticker]
        ax.plot(df_plot['Date'], df_plot['Cumulative_Return'], label=ticker)

    ax.set_title("Cumulative Return Over Time - Top 5 Performing Stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # -------- INSERT INTO SQL --------
    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM Cumulative_Return")
    if cursor.fetchone()[0] > 0:
        st.info("ℹ️ Cumulative Return already stored in database")
    else:
        for _, row in top5_df.iterrows():
            cursor.execute("""
                INSERT INTO Cumulative_Return
                (Date, Close, Ticker,Start_Close,Cumulative_Return,Cumulative_Return_Percent)
                VALUES (%s,%s,%s,%s,%s,%s)
            """, (row["Date"], row["Close"],row['Ticker'],row['Start_Close'],row['Cumulative_Return'],row['Cumulative_Return_Percent']))

        conn.commit()
        st.success("✅ Top 15 Cumulative Return stored in SQL")

    cursor.close()
    conn.close()
    
    return top5_tickers, cum_df

# -----------------------------------------------------------
# 3. Sector-wise Performance Page
# -----------------------------------------------------------

def avg_year_returns(folder_path, sector_file):
    # Load sector map and normalize
    if not os.path.exists(sector_file):
        st.error(f"Sector file not found: {sector_file}")
        return pd.Series(dtype=float)

    sector_map = pd.read_csv(sector_file)
    # Expect sector_map has columns like: Ticker, sector (case-insensitive). Normalize names:
    sector_map.columns = [c.strip() for c in sector_map.columns]
    # try to find ticker column name and sector column name
    possible_ticker_cols = [c for c in sector_map.columns if c.lower() in ('ticker', 'symbol', 'stock', 'company')]
    possible_sector_cols = [c for c in sector_map.columns if 'sector' in c.lower()]

    if not possible_ticker_cols or not possible_sector_cols:
        st.error("Sector CSV must contain a ticker column and a sector column (headers like 'Ticker' and 'sector').")
        return pd.Series(dtype=float)

    ticker_col = possible_ticker_cols[0]
    sector_col = possible_sector_cols[0]

    sector_map = sector_map[[ticker_col, sector_col]].rename(columns={ticker_col: 'Ticker', sector_col: 'sector'})
    sector_map['Ticker'] = sector_map['Ticker'].astype(str).str.upper().str.strip()

    # Read all stock CSVs and ensure Ticker column exists in each row
    output_folder = os.path.join(folder_path, "output_csv")
    all_files = glob.glob(os.path.join(output_folder, "*.csv"))

    all_daa = []
    for filename in all_files:
        ticker = os.path.splitext(os.path.basename(filename))[0].upper()
        df2 = pd.read_csv(filename)
        if 'date' not in df2.columns or 'close' not in df2.columns:
            continue
        df2['Ticker'] = ticker
        all_daa.append(df2[['Ticker', 'date', 'close']])

    if not all_daa:
        st.warning("No stock CSVs found to compute yearly returns.")
        return pd.Series(dtype=float)

    combined_df = pd.concat(all_daa, axis=0, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    combined_df = combined_df.sort_values(['Ticker', 'date'])

    # compute first and last price for each ticker (across available range)
    first_last = combined_df.groupby('Ticker').agg(
        first_price=('close', 'first'),
        last_price=('close', 'last')).reset_index()

    first_last['yearly_return'] = (first_last['last_price'] - first_last['first_price']) / first_last['first_price'] * 100

    merged = pd.merge(first_last, sector_map, on='Ticker', how='left')

    # If some tickers have no sector, mark them
    merged['sector'] = merged['sector'].fillna("Unknown")

    sector_perf = (
    merged.groupby('sector')['yearly_return']
    .mean()
    .reset_index()
    .sort_values(by='yearly_return', ascending=False))


    st.subheader("🏆 Average Yearly Return by Sector")
    st.write(sector_perf)

    fig, ax = plt.subplots(figsize=(12, 6))
    sector_perf.plot(kind='bar', ax=ax)
    ax.set_title("Average Yearly Return by Sector")
    ax.set_xlabel("Sector")
    ax.set_ylabel("Average Yearly Return (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # -------- INSERT INTO SQL --------

    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM Sector_wise")
    if cursor.fetchone()[0] > 0:
        st.info("ℹ️ Sector-wise Performance already stored in database")
    else:
        for _, row in sector_perf.iterrows():
            cursor.execute("""
                INSERT INTO Sector_wise
                (Sector, Yearly_Return)
                VALUES (%s,%s)
            """, (row["sector"], row["yearly_return"]))

        conn.commit()
        st.success("✅ Average Yearly Return by Sector is stored in SQL")

    cursor.close()
    conn.close()


    # return the Series for programmatic use
    return sector_perf

# -----------------------------------------------------------
# 4. Stock Price Correlation Page
# -----------------------------------------------------------
def stock_correlation(folder_path):
    output_folder = os.path.join(folder_path, "output_csv")
    all_files = glob.glob(os.path.join(output_folder, "*.csv"))

    stock_dict = {}

    for filename in all_files:
        df = pd.read_csv(filename)
        ticker = os.path.splitext(os.path.basename(filename))[0].upper()

        if 'date' not in df.columns or 'close' not in df.columns:
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').set_index('date')

        stock_dict[ticker] = pd.to_numeric(df['close'], errors='coerce')

    if not stock_dict:
        st.warning("No stock CSVs found for correlation.")
        return pd.DataFrame()

    price_df = pd.DataFrame(stock_dict)

    # ✅ USE RETURNS
    returns_df = price_df.pct_change().dropna()
    corr_matrix = returns_df.corr()

    # ---------------- DISPLAY ----------------
    st.subheader("🔍 STOCK RETURN CORRELATION MATRIX")
    st.dataframe(corr_matrix.round(3))

    fig, ax = plt.subplots(figsize=(12, 8))
    cax = ax.imshow(corr_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)

    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticklabels(corr_matrix.columns)

    plt.tight_layout()
    st.pyplot(fig)

    # ---------------- SQL INSERT ----------------
    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM Stock_Price_Correlation")
    conn.commit()

    for i, stock1 in enumerate(corr_matrix.columns):
        for j, stock2 in enumerate(corr_matrix.columns):
            if j >= i:
                cursor.execute("""
                    INSERT INTO Stock_Price_Correlation
                    (stock_1, stock_2, correlation)
                    VALUES (%s, %s, %s)
                """, (stock1, stock2, float(corr_matrix.iloc[i, j])))

    conn.commit()
    cursor.close()
    conn.close()

    st.success("✅ Stock return correlation matrix stored in SQL")

    return corr_matrix


# -----------------------------------------------------------
# 5. Top 5 Gainers and Losers (Month-wise) Page
# -----------------------------------------------------------

def monthly_gainers_losers(folder_path):

    output_folder = os.path.join(folder_path, "output_csv")
    all_files = glob.glob(os.path.join(output_folder, "*.csv"))

    all_data = []
    for file in all_files:
        ticker = os.path.splitext(os.path.basename(file))[0].upper()
        df = pd.read_csv(file)

        if 'date' not in df.columns or 'close' not in df.columns:
            continue

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['Ticker'] = ticker

        all_data.append(df[['Ticker', 'date', 'close']])

    if not all_data:
        st.warning("No CSVs found for monthly gainers/losers.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)

    # Extract month (YYYY-MM)
    combined_df['Month'] = combined_df['date'].dt.to_period('M').astype(str)

    # Monthly open → first close | Monthly close → last close
    monthly = (
        combined_df.sort_values('date')
        .groupby(['Ticker', 'Month'])['close']
        .agg(['first', 'last'])
        .reset_index()
    )

    # Monthly return
    monthly['Monthly_Return'] = ((monthly['last'] - monthly['first']) / monthly['first']) * 100

    # Available months
    unique_months = sorted(monthly['Month'].unique(), reverse=True)

    # Month selection
    selected_month = st.selectbox("Select Month", unique_months)

    df_month = monthly[monthly['Month'] == selected_month]

    if df_month.empty:
        st.warning("No data for selected month.")
        return

    # Top 5
    top5_gainers = df_month.nlargest(5, 'Monthly_Return')
    top5_losers = df_month.nsmallest(5, 'Monthly_Return')

    # ---------------------------- TABLES ---------------------------------

    st.subheader(f"📈 Top 5 Monthly Gainers – {selected_month}")
    st.dataframe(top5_gainers[['Ticker', 'Monthly_Return']])

    st.subheader(f"📉 Top 5 Monthly Losers – {selected_month}")
    st.dataframe(top5_losers[['Ticker', 'Monthly_Return']])

    # ---------------------------- CHARTS ---------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].barh(top5_gainers['Ticker'], top5_gainers['Monthly_Return'])
    axes[0].set_title(f"Top 5 Gainers - {selected_month}")
    axes[0].set_xlabel("Monthly Return (%)")

    axes[1].barh(top5_losers['Ticker'], top5_losers['Monthly_Return'],color='red')
    axes[1].set_title(f"Top 5 Losers - {selected_month}")
    axes[1].set_xlabel("Monthly Return (%)")

    plt.tight_layout()
    st.pyplot(fig)

    # -------- INSERT INTO SQL --------

    conn = get_connection("sector")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM Top_Gainers")
    if cursor.fetchone()[0] > 0:
        st.info("ℹ️ Top 5 Gainers already stored in database")
    else:
        for _, row in monthly.iterrows():
            cursor.execute("""
                INSERT INTO Top_Gainers
                (Ticker, Month,First,Last,Monthly_Return)
                VALUES (%s,%s,%s,%s,%s)
            """, (row["Ticker"],row['Month'],row['first'],row['last'],row['Monthly_Return']))

        conn.commit()
        st.success("✅ Top 5 Gainers is stored in SQL")

    cursor.close()
    conn.close()



# -----------------------------------------------------------
# UI PAGES
# -----------------------------------------------------------
def home():
    st.title("📊 Data-Driven Stock Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DATA SETUP")
        if st.button("🗄️ Setup Database"):
            setup_database()

        if st.button("📥 Load YAML & Insert"):
            df = load_yaml_data()
            insert_into_db(df)

        if st.button("📁 Export CSV"):
            export_csv()
        
    with col2:
        st.subheader("📈 VISUALIZATION")
        if st.button("Volatility Analysis"):
            st.session_state.page = "volatility"
            st.rerun()
        
        if st.button("2. Cumulative Return Over Time"):
            st.session_state.page = "cumulative"
            st.rerun()

        if st.button("3. Sector-wise Performance"):
            st.session_state.page = "year_return"
            st.rerun()

        if st.button("4. Stock Price Correlation"):
            st.session_state.page = "stock_correlation"
            st.rerun()

        if st.button("5. Top 5 Gainers and Losers (Month-wise)"):
            st.session_state.page = "gainers_lose"
            st.rerun()

# -----------------------------------------------------------
# 1. Volatility Analysis Page
# -----------------------------------------------------------

def volatility_page():
    st.title("📈 Top 10 Volatile Stocks")

    vol = calculate_and_store_volatility()
    if vol.empty:
        return

    st.dataframe(vol)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(vol["Ticker"], vol["Volatility"])
    ax.set_ylabel("Volatility")
    ax.set_title("Top 10 Volatile Stocks")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if st.button("⬅ Back"):
        st.session_state.page = "home"
        st.rerun()

# -----------------------------------------------------------
# 2. Cumulative Return Over Time Page
# -----------------------------------------------------------

def show_cumulative():
    st.title("📈 Top 5 Cumulative Returns")
    top5, cum_df = cumulative_return_plot(folder_path)
    if not top5:
        st.info("No cumulative data available. Ensure you exported CSVs into output_csv folder.")
    if st.button("⬅ Go Back"):
        st.session_state.page = "home"
        st.rerun()

# -----------------------------------------------------------
# 3. Sector-wise Performance Page
# -----------------------------------------------------------

def show_year_return():
    st.title("📊 Average Yearly Return by Sector")
    sector_perf = avg_year_returns(folder_path, sector_file)
    if sector_perf is None or sector_perf.empty:
        st.info("No sector performance to show.")
    if st.button("⬅ Go Back"):
        st.session_state.page = "home"
        st.rerun()


# -----------------------------------------------------------
# 4. Stock Price Correlation Page
# -----------------------------------------------------------

def show_stock_corr():
    st.title("🔗 Stock Price Correlation")
    corr = stock_correlation(folder_path)
    if corr is None or corr.empty:
        st.info("No correlation data to show.")
    if st.button("⬅ Go Back"):
        st.session_state.page = "home"
        st.rerun()

# -----------------------------------------------------------
# 5. Top 5 Gainers and Losers (Month-wise) Page
# -----------------------------------------------------------

def show_gainers_lose():
    st.title("📅 Monthly Gainers & Losers")
    monthly_gainers_losers(folder_path)
    if st.button("⬅ Go Back"):
        st.session_state.page = "home"
        st.rerun()

# -----------------------------------------------------------
# ROUTER
# -----------------------------------------------------------
if st.session_state.page == "home":
    home()
elif st.session_state.page == "volatility":
    volatility_page()
elif st.session_state.page == "cumulative":
    show_cumulative()
elif st.session_state.page == "year_return":
    show_year_return()
elif st.session_state.page == "stock_correlation":
    show_stock_corr()
elif st.session_state.page == "gainers_lose":
    show_gainers_lose()
else:
    st.session_state.page = "home"
    st.rerun()
