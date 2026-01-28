# 📊 Data-Driven Stock Analysis 

This project is a data-driven stock market analysis application built using Python, SQL, and Streamlit.
It focuses on organizing raw stock data, cleaning it, storing it in a relational database, and generating insightful visualizations to analyze market behavior, sector performance, volatility, correlations, and monthly trends.

The system transforms YAML-based historical stock data into structured datasets, performs financial analytics, and presents results through an interactive Streamlit dashboard.

## **🔨 Development Process**

### **1. Data Collection**

   * Stock price data stored as monthly YAML files
   * Sector mapping provided via CSV

### **2. Data Cleaning & Transformation**

   * Parsed YAML files into Pandas DataFrames
   * Cleaned missing values and standardized date formats

### **3. Database Design**

   * Created MySQL database and normalized tables
   * Stored processed analytical outputs for reusability

### **4. Analysis & Computation**

   * Volatility calculation using daily returns
   * Cumulative returns, sector-wise performance
   * Correlation analysis using stock returns
   * Monthly gainers and losers computation

### **5. Visualization**

   * Interactive Streamlit UI
   * Matplotlib-based charts and tables

## **✨ Key Features**

### **📈 Volatility Analysis**
   * Identifies the top 10 most volatile stocks using daily return standard deviation.

### **📊 Cumulative Return Tracking**
   * Visualizes long-term stock performance by tracking cumulative returns over time.

### **🏭 Sector-wise Performance Comparison**
   * Compares average yearly returns across market sectors to highlight outperforming industries.

### **🔗 Stock Return Correlation Matrix**
   * Analyzes relationships between stocks using return-based correlation to assess diversification potential.

### **📅 Monthly Top Gainers & Losers**
   * Displays the top 5 gaining and losing stocks each month based on percentage returns.

## **⚙️ Tech Stack**

* Programming Language: Python
* Data Processing: Pandas, NumPy
* Visualization: Matplotlib
* Database: MySQL
* Web Framework: Streamlit
* File Formats: YAML, CSV
* Version Control: Git & GitHub

## **📋 Project Overview**

This application converts raw market data into meaningful financial insights by:

  * Structuring unorganized stock data
  * Performing quantitative market analysis
  * Visualizing trends and relationships
  * Allowing users to explore stock behavior interactively

It is designed for data analysis, financial analytics, and visualization learning projects.
## 🔥 Features
### **🗄️ Database Setup & Management**
   * Automated creation and management of MySQL databases and tables for structured storage.

### **🔄 YAML to SQL Data Pipeline**
   * Converts raw YAML stock data into cleaned, structured SQL records.

### MySQL Integration

  * Create database (DATA)
  * Create table (MARKET)
  * Insert all YAML records into MySQL
  * Export each stock’s data into individual CSV files

### Multi-Page Streamlit Dashboard

Includes the following analytics pages:

#### 1. Volatility Analysis

* Computes daily return volatility
* Displays Top 10 Most Volatile Stocks and Storing that to data Base.

#### 2. Cumulative Return Analysis

   * Calculates cumulative return from the starting point
   * Plots Top 5 Stocks over time and Storing that to data Base.

#### 3. Sector-wise Performance

   * Reads sector mapping CSV
   * Computes yearly return for each stock
   * Plots average yearly return per sector and Storing that to data Base.

#### 4. Stock Price Correlation

   * Correlation heatmap of stock closing pricesand Storing that to data Base.

#### 5. Top 5 Gainers & Losers (Month-wise)

   * Computes monthly returns
   * Shows the best and worst performing stocks per month and Storing that to data Base.
### 🎛️ User-friendly Streamlit Navigation
   * Interactive dashboard with intuitive controls for seamless analysis.

## 📁 Project Structure
```
📂 Project Root
│── app.py                     # Streamlit main application
│── requirements.txt           # Required Python packages
│── Sector_data.csv            # Stock-to-sector mapping file
│
📂 data
│   ├── 2023-10
│   │     ├── *.yaml           # Monthly stock data
│   ├── 2023-11
│   │     ├── *.yaml
│   ├── ...
│   ├── output_csv/            # Auto-generated CSV files per stock

```

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```
### 2️⃣ Install Dependencies

Create a virtual environment (recommended):

```pip install -r requirements.txt```

If you don’t have a requirements.txt, use:

```pip install streamlit pandas numpy matplotlib mysql-connector-python pyyaml```

### 3️⃣ Update File Paths

Edit these paths inside the code:

```
folder_path = r"D:\PROJECT_2\Data-Driven-Stock-Analysis\data"
sector_file = r"D:\PROJECT_2\Data-Driven-Stock-Analysis\Sector_data - Sheet1.csv"
```

### 4️⃣ Setup MySQL

Run MySQL server and create database:

`CREATE DATABASE DATA;`


OR let the app create it automatically using:

```
create_database()
create_table()
```

Update MySQL credentials in:

```
user="root",
password="0007"
```
▶️ Run the Streamlit App

`streamlit run app.py`

💡 After launch, the sidebar shows 5 analysis tools.

## 🔄 How It Works

* Load YAML stock data
* Insert cleaned data into MySQL
* Export structured CSV files
* Perform financial analysis:
   * Volatility
   * Returns
   * Correlation
* Visualize insights using Streamlit
* Store results back into the database

## **🎯 Use Case**

  📌 Financial data analysis practice
  
  📌 Learning SQL + Python integration
  
  📌 Market trend exploratio
  
  📌 Data visualization projects
  
  📌 Portfolio project for Data Analyst / Data Scientist roles

## **🚀 Future Enhancements**

   📉 Risk-adjusted metrics (Sharpe Ratio, Beta)
   
   📊 Interactive Plotly visualization
   
   ☁️ Cloud database integration
   
   🧠 Machine learning-based stock prediction 
   
   🔐 User authentication & dashboards
   
   📈 Real-time stock data API integration
