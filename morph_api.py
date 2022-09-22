import requests

class MorphoDiTa:
    """
    Class calling the MorphoDiTa API.
    """
    def __init__(self):
        self.url = "http://lindat.mff.cuni.cz/services/morphodita/api/"
        self.params = {
            "output": "json"
        }

    def tag(self, text):
        """
        Query MorphoDiTa for the morphological tags of a text.
        """
        self.params["data"] = text
        r = requests.get(self.url + "tag", params=self.params)
        return r.json()["result"]

    def generate(self, lemma):
        """
        Query MorphoDiTa for all possible forms of a lemma.
        """
        self.params["data"] = lemma
        r = requests.get(self.url + "generate", params=self.params)
        return r.json()["result"]

if __name__ == "__main__":
    mt = MorphoDiTa()
    print(mt.tag("Tohle je hračka . Bylo nás pět ."))
    print(mt.generate("hračka"))