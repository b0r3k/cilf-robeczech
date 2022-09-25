import requests
from typing import List, Dict

class MorphoDiTa:
    """
    Class calling the MorphoDiTa API.
    """
    def __init__(self):
        self.url = "http://lindat.mff.cuni.cz/services/morphodita/api/"
        self.params = {
            "output": "json"
        }
        self.lemma_params = {
            "convert_tagset": "strip_lemma_id",
            "output": "json"
        }

    def tag(self, text: str) -> List[List[Dict]]:
        """
        Query MorphoDiTa for the morphological tags of a text.
        """
        self.params["data"] = text
        r = requests.get(self.url + "tag", params=self.params)
        return r.json()["result"]

    def generate(self, lemma: str) -> List[List[Dict]]:
        """
        Query MorphoDiTa for all possible forms of a lemma.
        """
        self.params["data"] = lemma
        r = requests.get(self.url + "generate", params=self.params)
        return r.json()["result"]

    def lemmatize(self, word: str) -> str:
        """
        Query MorphoDiTa for the lemma of a word.
        """
        self.lemma_params["data"] = word
        r = requests.get(self.url + "tag", params=self.lemma_params)
        return r.json()["result"][0][0]["lemma"]

if __name__ == "__main__":
    mt = MorphoDiTa()
    print(mt.tag("Tohle je hračka . Bylo nás pět ."), "\n")
    print(mt.generate("hračka"), "\n")
    print(mt.lemmatize("bílá"))
