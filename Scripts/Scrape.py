import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to scrape a single page
def scrape_page(url):
    # Send a GET request to the website
    response = requests.get(url)
    
    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table with id 'custom-table'
    table = soup.find('table', {'id': 'custom-table'})

    # List to store extracted data
    data = []

    # Loop through the rows in the table (skip the header)
    for row in table.find_all('tr', {'class': 'alert'}):
        try:
            image_url = row.find('td', {'class': 'image'}).find('a').get('href')
        except AttributeError:
            image_url = None

        try:
            category = row.find('td', {'class': 'description'}).find('strong').text.strip()
        except AttributeError:
            category = None

        try:
            description = row.find('td', {'class': 'description'}).find('label').text.strip()
        except AttributeError:
            description = None

        try:
            sold_as = row.find('td', {'class': 'soldas'}).text.strip()
        except AttributeError:
            sold_as = None

        try:
            result_rows = row.find('td', {'class': 'result'}).find_all('tr')
            results = ', '.join([f"{r.find('td').text.strip()}:{r.find_all('td')[1].text.strip()}" for r in result_rows if len(r.find_all('td')) > 1])
        except AttributeError:
            results = None

        try:
            fentanyl = row.find('td', {'class': 'fentstrip'}).text.strip()
        except AttributeError:
            fentanyl = None

        try:
            notes = row.find('td', {'class': 'notes'}).text.strip()
        except AttributeError:
            notes = None

        try:
            location = row.find('td', {'class': 'location'}).text.strip()
        except AttributeError:
            location = None

        try:
            date = row.find('td', {'class': 'date'}).text.strip()
        except AttributeError:
            date = None

        # Append the extracted data to the list
        data.append({
            'Image': image_url,
            'Category': category,
            'Description': description,
            'Sold as': sold_as,
            'Results': results,
            'Fentanyl & Benzodiazepine Test': fentanyl,
            'Notes': notes,
            'Location': location,
            'Date': date
        })

    return data

# Function to scrape all pages
def scrape_all_pages(base_url):
    all_data = []

    # Send a GET request to get the total number of pages
    response = requests.get(base_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the pagination element and extract the total number of pages
    pagination = soup.find('div', {'class': 'pagination'})
    total_pages = int(pagination.find_all('a')[-1].text)  # Get the last page number

    print(f"Total pages: {total_pages}")

    # Loop through all pages
    for page_num in range(1, total_pages + 1):
        url = f"{base_url}/page/{page_num}/"
        print(f"Scraping page {page_num}...")

        # Scrape the current page
        data = scrape_page(url)

        # Add the data from this page to the overall data list
        all_data.extend(data)
    
    return all_data

# Base URL without the page number
base_url = 'https://getyourdrugstested.com/alerts'

# Scrape all pages
all_data = scrape_all_pages(base_url)

# Convert the list to a DataFrame
df = pd.DataFrame(all_data)

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
df.to_csv('../Data/Processed/scraped_data.csv', index=False)

# Confirm that the file has been saved
print("Data saved to 'scraped_data.csv'")