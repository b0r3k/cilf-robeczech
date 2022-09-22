import pandas as pd 
import requests 
from bs4 import BeautifulSoup

colors_url = "https://cs.wikipedia.org/wiki/Seznam_barev"

class WikiScraper:
    def __init__(self, url: str) -> None:
        """Scrape table from Wikipedia page into a Pandas Dataframe."""
        # Sent GET request to the Wikipedia URL
        response = requests.get(url)

        # Parse data from the html into a Beautifulsoup object
        soup = BeautifulSoup(response.text, 'html.parser')
        indiatable = soup.find('table', {'class':'wikitable'})

        # Convert Wikipedia Table into a Python Dataframe
        df = pd.read_html(str(indiatable))
        self.df = pd.DataFrame(df[0])
    
    def filter_data(self, column_name: str) -> None:
        """Extract data from selected column and filter out multi-word expressions."""
        self.df = self.df[column_name]
        self.df = self.df.loc[~self.df.str.contains(" ")]

    def save_table(self, json_name: str) -> None:
        """Save the table to a JSON file."""
        self.df.to_json(json_name, orient="records", force_ascii=False, indent=4)

ws = WikiScraper(colors_url)
ws.filter_data("Barva")
ws.save_table("lexicons/colors.json")