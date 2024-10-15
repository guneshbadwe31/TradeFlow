# TradeFlow: Stock Trend Prediction Using Machine Learning

TradeFlow is a machine learning-based project designed to predict stock market trends with enhanced accuracy. It analyzes historical stock data, scrapes data from Yahoo Finance, and leverages deep learning models to forecast future price movements, providing valuable insights for traders and investors.

## Features

- **Data Scraping:** Fetches historical stock data from Yahoo Finance using `yfinance`.
- **Trend Study:** Analyzes the stock's trend over the last five years, displaying key metrics such as:
  - Opening, closing, high, and low values for the past five years.
- **Visualizations:**
  - **Closing Value vs. Time Charts:**
    - Last 5 years
    - Last 365 days
    - Last 30 days
  - **Stock Trend Prediction:**
    - Forecasts the percentage return the stock will provide over the next five years.
- **Predictive Modeling:** Uses the Long Short-Term Memory (LSTM) model built with `Keras` and powered by `TensorFlow`.
- **Interactive Visualizations:** Provides dynamic visualizations using `Plotly`.
- **Web Interface:** Displays output through a user-friendly interface powered by `Streamlit`.

## Technologies Used

- **Programming Languages:** Python
- **Libraries/Frameworks:**
  - `Keras`
  - `TensorFlow`
  - `NumPy`
  - `pandas`
  - `yfinance`
  - `pandas_datareader`
  - `Plotly`
  - `Streamlit`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TradeFlow.git
   cd TradeFlow
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Data Collection:**
   - Fetch stock market data using `yfinance` to retrieve the last five years of stock data.

2. **Run the Application:**
   - Start the Streamlit app to display predictions and charts in your browser:
     ```bash
     streamlit run app.py
     ```

3. **Data Visualization:**
   - View the stock’s opening, closing, high, and low values for the last five years.
   - Explore interactive charts for the stock’s closing values over different time ranges (last 5 years, last 365 days, last 30 days).
   
4. **Prediction:**
   - TradeFlow forecasts the potential percentage return the stock will provide over the next five years.


## Contributing

Contributions are welcome! If you’d like to contribute, please fork the repository and make your changes via a pull request. You can also open an issue for bug reports or feature suggestions.

