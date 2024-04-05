import pandas as pd
import matplotlib.pyplot as plt
import os

# Directory containing your CSV files
directory = 'Data/processedData3'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
all_data = []

# Iterate over each CSV file
count=0
for file in csv_files:
    # Read the CSV file

    df = pd.read_csv(os.path.join(directory, file))
    # Append the DataFrame to the list
    all_data.append(df)
    print(count)
    count+=1

# Concatenate all DataFrames in the list into a single DataFrame
combined_data = pd.concat(all_data, ignore_index=True)

# Assuming your CSV files have columns named 'column1' and 'column2', replace them with actual column names if different
data1 = combined_data['Score']


# Create subplots for each distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(data1, bins=40, color='blue', alpha=0.7)
plt.title('Distribution of Column 1')

plt.tight_layout()
plt.show()