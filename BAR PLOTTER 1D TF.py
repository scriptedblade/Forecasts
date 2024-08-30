import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo

# Load the CSV file
file_path = r'C:\Users\dinnu\Desktop\11.csv'
df = pd.read_csv(file_path)

# Convert 'Time' column to datetime with the correct format
df['Time'] = pd.to_datetime(df['Time'], format='%d-%m-%Y %H:%M:%S %z')

# Set 'Time' column as the index
df.set_index('Time', inplace=True)

# Ensure the data is sorted by time
df = df.sort_index()

# Resample the data to daily intervals and drop missing data
df_resampled = df['Last'].resample('D').ohlc().dropna()
df_resampled['Volume'] = df['Size'].resample('D').sum().dropna()

# Reset the index to use 'Time' as a column again
df_resampled.reset_index(inplace=True)

# Convert the Time column to string to use it as categorical data
df_resampled['Time'] = df_resampled['Time'].dt.strftime('%Y-%m-%d')

# Plot the data using Plotly
fig = go.Figure(data=[
    go.Candlestick(
        x=df_resampled['Time'],
        open=df_resampled['open'],
        high=df_resampled['high'],
        low=df_resampled['low'],
        close=df_resampled['close'],
        name='Candlestick',
        text=df_resampled.apply(lambda row: f"Volume: {row['Volume']}", axis=1),
        hoverinfo='x+text'
    )
])

# Customize the layout to match the style of the provided chart
fig.update_layout(
    title='Tick Data Resampled to Daily Intervals',
    xaxis_title='Time',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,  # Enable range slider
    xaxis=dict(type='category'),  # Set x-axis type to category
    hovermode='x',  # Enable crosshair cursor for better interactivity
    dragmode='pan',  # Enable panning
)

# Save the plot as an offline HTML file
output_file_path = r'C:\Users\dinnu\Desktop\Tick_Data_Daily_Intervals.html'
pyo.plot(fig, filename=output_file_path, auto_open=False)

print(f"Plot saved to {output_file_path}")
