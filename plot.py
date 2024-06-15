import pandas as pd
import matplotlib.pyplot as plt

# Function to read CSV file
def read_csv(file_path):
    return pd.read_csv(file_path)

# Function to plot data with smoothing
def plot_data_with_smoothing(df, output_file, window_size=10):
    # Apply rolling average for smoothing
    df['Smoothed_Value'] = df['Value'].rolling(window=window_size, min_periods=1).mean()
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Step'], df['Value'], label='Original Value', color='blue', alpha=0.2)
    plt.plot(df['Step'], df['Smoothed_Value'], label='Smoothed Value', color='red')
    plt.title('Success Rate over Steps with Smoothing in Lava scenario of MiniGrid')
    plt.xlabel('Steps')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)  # Save the plot as an image file
    plt.show()

# Main script
csv_file = 'success_rate_lava.csv'  # Replace with the path to your CSV file
output_file = 'success_rate_lava.png'  # File name for the saved plot

# Read the data from the CSV file
df = read_csv(csv_file)

# Plot the data with smoothing and save the plot
plot_data_with_smoothing(df, output_file)
