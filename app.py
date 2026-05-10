from flask import Flask, json, jsonify, request, render_template
import pandas as pd
import numpy as np
from keras.models import load_model
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route("/")
def mm():
    return render_template("home.html")

@app.route("/main1")
def main1():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template('register.html')

def get_stock_suggestions():
    with open("stocks.json", "r") as file:
        data = json.load(file)
    return data




@app.route("/home", methods=["GET", "POST"])
def home():
    stock = "MSFT"
    if request.method == "POST":
        stock = request.form["stock"]

    # Load stock suggestions
    stock_suggestions = get_stock_suggestions()

    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)

    # Fetch historical data
    google_data = yf.download(stock, start, end)

    # Fetch latest data for live tracking
    latest_data = yf.download(stock, period="1d", interval="1m")
    latest_close = latest_data['Close'].iloc[-1] if not latest_data.empty else None

    model = load_model("Latest_stock_price_model.keras")

    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data.Close[splitting_len:])

    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(x_test[['Close']])

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    predictions = model.predict(x_data)

    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Prepare plotting_data
    plotting_data = pd.DataFrame(
        {
            'Original Values': inv_y_test.reshape(-1),
            'Predicted Values': inv_pre.reshape(-1)
        },
        index=google_data.index[splitting_len + 100:]
    )
    
    # Reverse the order for plotting
    google_data = google_data[::-1]
    plotting_data = plotting_data[::-1]

    # Create combined data for plotting
    combined_data = pd.concat([google_data[['Close']], plotting_data[['Predicted Values']]], axis=1)

    def create_plot(data, title, y1, y2=None):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=data.index, y=data[y1], name=y1, mode='lines'))
        if y2:
            fig.add_trace(go.Scatter(x=data.index, y=data[y2], name=y2, mode='lines'))
        fig.update_layout(title_text=title)
        return fig.to_html(full_html=False)

    ma_250_plot = create_plot(google_data, 'Original Close Price and MA for 250 days', 'Close', 'MA_for_250_days')
    ma_200_plot = create_plot(google_data, 'Original Close Price and MA for 200 days', 'Close', 'MA_for_200_days')
    ma_100_plot = create_plot(google_data, 'Original Close Price and MA for 100 days', 'Close', 'MA_for_100_days')
    ma_100_250_plot = create_plot(google_data, 'Original Close Price and MA for 100 days and MA for 250 days', 'MA_for_100_days', 'MA_for_250_days')

    close_price_plot = create_plot(combined_data, 'Original Close Price vs Predicted Close price', 'Close', 'Predicted Values')

    # Create a plot for live tracking of the original close price
    live_tracking_plot = create_plot(latest_data, 'Live Tracking of Original Close Price', 'Close')

    return render_template("index.html", stock=stock, tables=[google_data.to_html(classes='data')],
                           ma_250_plot=ma_250_plot, ma_200_plot=ma_200_plot, ma_100_plot=ma_100_plot,
                           ma_100_250_plot=ma_100_250_plot, close_price_plot=close_price_plot,
                           plotting_data=plotting_data.to_html(classes='data', index=False),
                           stock_suggestions=json.dumps(stock_suggestions),
                           latest_close=latest_close,
                           live_tracking_plot=live_tracking_plot)


@app.route("/search_suggestions", methods=["GET"])
def search_suggestions():
    query = request.args.get('query', '')
    suggestions = []
    
    if query:
        all_suggestions = get_stock_suggestions()
        for item in all_suggestions:
            if item['Name'] and query.lower() in item['Name'].lower():
                suggestions.append({
                    'StockID': item['StockID'],
                    'Name': item['Name'],
                    'Country': item['Country'],
                    'Category Name': item['Category Name']
                })
    
    return jsonify(suggestions)

@app.route("/FuturePrediction", methods=["GET", "POST"])
def FuturePrediction():
    stock = "GOOG"
    if request.method == "POST":
        stock = request.form["stock"]

    # Load stock suggestions
    stock_suggestions = get_stock_suggestions()     

    end = datetime.now()
    start = datetime(end.year - 20, end.month, end.day)
    stock_data = yf.download(stock, start, end)

    Adj_close_price = stock_data[['Adj Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(Adj_close_price)

    x_data = []
    y_data = []

    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i - 100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Load pre-trained model
    model = load_model("Future_stock_price_model.keras")

    def predict_future(model, last_100_days, days_to_predict):
        predicted_prices = []
        current_input = last_100_days

        for _ in range(days_to_predict):
            prediction = model.predict(current_input.reshape(1, -1, 1))
            predicted_prices.append(prediction[0][0])

            current_input = np.append(current_input[1:], prediction[0][0])
            current_input = current_input.reshape(-1, 1)

        return np.array(predicted_prices)

    days_to_predict = 30
    last_100_days = scaled_data[-100:]
    predicted_prices = predict_future(model, last_100_days, days_to_predict)
    predicted_prices = scaler.inverse_transform(predicted_prices.reshape(-1, 1))

    future_dates = pd.date_range(start=stock_data.index[-1], periods=days_to_predict + 1)[1:]
    predicted_prices_df = pd.DataFrame(predicted_prices, index=future_dates, columns=["Predicted Price"])

    # Create interactive plot for historical and predicted data
    historical_trace = go.Scatter(
        x=stock_data.index,
        y=stock_data['Adj Close'],
        mode='lines',
        name='Historical Data'
    )

    predicted_trace = go.Scatter(
        x=predicted_prices_df.index,
        y=predicted_prices_df['Predicted Price'],
        mode='lines',
        name='Future Predictions',
        line=dict(color='orange')
    )

    fig_graph = go.Figure(data=[historical_trace, predicted_trace])
    fig_graph.update_layout(
        title=f"{stock} Stock Price Prediction",
        xaxis_title='Date',
        yaxis_title='Adj Close Price',
        hovermode='x',
        height=600
    )

    graph_html = fig_graph.to_html(full_html=False, include_plotlyjs='cdn')

    # Create interactive table
    table_trace = go.Table(
        header=dict(values=["Date", "Predicted Price"], fill_color='paleturquoise', align='left'),
        cells=dict(values=[predicted_prices_df.index.date, predicted_prices_df['Predicted Price']],
                   fill_color='lavender', align='left')
    )

    fig_table = go.Figure(data=[table_trace])
    fig_table.update_layout(
        height=400
    )

    table_html = fig_table.to_html(full_html=False, include_plotlyjs='cdn')

    return render_template("FuturePrediction.html", stock=stock, graph=graph_html, table=table_html,
                           stock_suggestions=json.dumps(get_stock_suggestions()))

@app.route("/latest_stock_data", methods=["GET"])
def latest_stock_data():
    stock = request.args.get("stock", "MSFT")  # Default to MSFT if no stock is provided
    end = datetime.now()
    start = end - pd.DateOffset(days=1)  # Fetch data for the last 1 day

    # Fetch the latest stock data
    stock_data = yf.download(stock, start=start, end=end)

    # Extract today's data if available, otherwise use the most recent data
    today_data = stock_data.tail(1) if not stock_data.empty else None
    
    if today_data is None or today_data.empty:
        start = end - pd.DateOffset(days=30)  # Adjust this range as needed
        stock_data = yf.download(stock, start=start, end=end)
        today_data = stock_data.tail(1) if not stock_data.empty else None

    # Additional information
    stock_info = yf.Ticker(stock).info

    if today_data is not None and not today_data.empty:
        today_data = today_data.reset_index()
        latest_data_html = today_data.to_html(classes='data', index=False)
        
        # Collect additional information
        additional_data = {
            'Close': stock_info.get('regularMarketPrice', 'N/A'),
            'Open': stock_info.get('regularMarketOpen', 'N/A'),
            'High': stock_info.get('regularMarketDayHigh', 'N/A'),
            'Low': stock_info.get('regularMarketDayLow', 'N/A'),
            'Volume': stock_info.get('regularMarketVolume', 'N/A'),
            'Previous Close': stock_info.get('regularMarketPreviousClose', 'N/A'),
            'Bid': stock_info.get('bid', 'N/A'),
            'Ask': stock_info.get('ask', 'N/A'),
            'Day\'s Range': f"{stock_info.get('dayLow', 'N/A')} - {stock_info.get('dayHigh', 'N/A')}",
            '52 Week Range': f"{stock_info.get('fiftyTwoWeekLow', 'N/A')} - {stock_info.get('fiftyTwoWeekHigh', 'N/A')}",
            'Avg. Volume': stock_info.get('averageVolume', 'N/A'),
            'Market Cap': stock_info.get('marketCap', 'N/A'),
            'Beta': stock_info.get('beta', 'N/A'),
            'PE Ratio': stock_info.get('forwardEps', 'N/A'),
            'EPS': stock_info.get('trailingEps', 'N/A'),
            'Earnings Date': stock_info.get('earningsDate', 'N/A'),
            'Forward Dividend & Yield': f"{stock_info.get('forwardDividendRate', 'N/A')} ({stock_info.get('forwardDividendYield', 'N/A')})",
            'Ex-Dividend Date': stock_info.get('exDividendDate', 'N/A'),
            '1y Target Est': stock_info.get('targetMeanPrice', 'N/A')
        }

        additional_data_html = '<table class="data">'
        additional_data_html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>'
        for key, value in additional_data.items():
            additional_data_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
        additional_data_html += '</tbody></table>'
    else:
        latest_data_html = "<p>No data available.</p>"
        additional_data_html = "<p>No additional data available.</p>"

    return jsonify({
        'latest_data': latest_data_html,
        'additional_data': additional_data_html
    })

if __name__ == "__main__":
    app.run(debug=True)
