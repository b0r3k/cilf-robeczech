import pandas as pd 
import requests 
from bs4 import BeautifulSoup

wiki_url = "https://cs.wikipedia.org/wiki/Seznam_barev"
table_class = "wikitable sortable jquery-tablesorter"

# Sent GET request to the Wikipedia URL
response = requests.get(wiki_url)

# Parse data from the html into a Beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
indiatable = soup.find('table', {'class':"wikitable"})

# Convert Wikipedia Table into a Python Dataframe
df = pd.read_html(str(indiatable))
df = pd.DataFrame(df[0])

# Clean the Data
colors = df['Barva']
colors = colors.loc[~colors.str.contains(" ")]

# Save common surnames to JSON file
colors.to_json("lexicons/common_colors.json", orient="records", force_ascii=False, indent=4)