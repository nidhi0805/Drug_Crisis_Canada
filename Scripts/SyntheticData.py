import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL of the page
url = 'https://www.canada.ca/en/health-canada/services/opioids/data-surveillance-research/homelessness-substance-related-acute-toxicity-deaths.html'

# Send GET request to fetch the HTML content
response = requests.get(url)

# Parse the page using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Find the table by its ID
table = soup.find('table', id='tbl1')

# Extract headers (consider the first row as the header)
headers = []
for th in table.find_all('th'):
    headers.append(th.get_text(strip=True))

# Extract the rows (skip the first header row)
rows = []
for row in table.find_all('tr')[1:]:  # Skip the first row (headers)
    cols = row.find_all('td')
    # If columns are not empty, extract text for each column
    if len(cols) > 0:
        data = [col.get_text(strip=True) for col in cols]
        rows.append(data)

# Create a DataFrame from the rows and columns
df = pd.DataFrame(rows, columns=headers)

# Print the first few rows to verify the structure
print(df.head())

# Save the DataFrame to a CSV file
df.to_csv('homelessness_overdose_data.csv', index=False)

print(f"Table data extracted and saved to 'homelessness_overdose_data.csv'")
