from collections import defaultdict
from glob import glob
from json import load
from morph_api import MorphoDiTa
import numpy as np
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
from pathlib import Path

class Editor:
    def __init__(self):
        self.lexicons = self._load_lexicons()
        self.morphodita = MorphoDiTa()
        
        model_checkpoint = "ufal/robeczech-base"
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def _load_lexicons(self):
        """ Load lexicons from the lexicons directory. """
        lexicons = {}
        for f_name in glob('lexicons/*.json'):
            with open(f_name, 'r') as json_f:
                lexicons[Path(f_name).stem] = load(json_f)
        return lexicons

    def template(self, template, word):
        """
        Fill the template with the given word in the correct form.

        Parameters:
            template (str): str with one [MASK] token
            word (str): word to be masked
        
        Returns:
            str: template with [MASK] replaced by word in correct form
        """
        # Get the possible forms of the word
        generated = self.morphodita.generate(word)[0]
        tag_words = defaultdict(list)
        for gen in generated:
            tag_words[gen["tag"]].append(gen["form"])

        # Use language model to predict words suitable for given context
        inputs = self.tokenizer(template, return_tensors="np")
        token_logits = self.model(**inputs).logits
        # Find the location of [MASK] and extract its logits
        mask_token_index = np.argwhere(inputs["input_ids"] == self.tokenizer.mask_token_id)[0, 1]
        mask_token_logits = token_logits[0, mask_token_index, :]
        # Pick the [MASK] candidates with the highest logits
        # We negate the array before argsort to get the largest, not the smallest, logits
        top_tokens = np.argsort(-mask_token_logits)[:50].tolist()
        tokens = template.split()
        mask_index = tokens.index("[MASK]")
        sentences = []
        for result in top_tokens:
            tokens[mask_index] = self.tokenizer.decode([result]).strip()
            sentences.append(" ".join(tokens))
        sentences_joined = "\n".join(sentences)

        # Get the possible tags from predicted words
        tagged = self.morphodita.tag(sentences_joined)
        tags = []
        for sentence in tagged:
            tag = sentence[mask_index]["tag"]
            if tag in tag_words:
                tags.append(tag)
                # print(tag, "\t", sentence[mask_index]["token"])
        
        # Find the most frequent tag in a tag list
        most_freq_tag = max(set(tags), key = tags.count)
        for form in tag_words[most_freq_tag]:
            print(template.replace("[MASK]", form))

if __name__ == "__main__":
    e = Editor()
    print(e.lexicons.keys())
    e.template("V [MASK] je krásně.", "příroda")