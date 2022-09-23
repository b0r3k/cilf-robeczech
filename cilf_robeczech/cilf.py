from collections import defaultdict
from glob import glob
from json import load
from cilf_robeczech.morph_api import MorphoDiTa
import numpy as np
import os
from transformers import TFAutoModelForMaskedLM, AutoTokenizer
from pathlib import Path

class Editor:
    """
    Class for generating Czech sentences from templates.
    Templates can be filled in multiple ways 
        – providing a list of words to be filled in,
        – specifying a predefined lexicon to take the words from, 
        – letting the Masked Language Model provide suggestions.
    """
    def __init__(self):
        """
        Load the RobeCzech model, download from HF if necessary. 
        Load all the lexicons into memory.
        Create a class to simplify the communication with MorphoDiTa API.
        """
        self.lexicons = self._load_lexicons()
        self.morphodita = MorphoDiTa()
        
        model_checkpoint = "ufal/robeczech-base"
        self.model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.spec_tokens_mapping = {
            "[FORMAT]": None, "[MASK]": None, "[DEN]": "days", "[MESIC]": "months", 
            "[MUZ_JMENO]": "male_names", "[ZENA_JMENO]": "female_names", 
            "[MESTO]": "cities", "[BARVA]": "colors", "[MUZ_PRIJMENI]": "male_surnames", 
            "[ZENA_PRIJMENI]": "female_surnames"
        }

    def list_special_tokens(self):
        """
        Return valid special tokens for the templates.
        """
        return list(self.spec_tokens_mapping.keys())

    def _load_lexicons(self):
        """ 
        Load all lexicons from the lexicons directory into memory. 
        """
        lexicons = {}
        for f_name in glob(os.path.join(os.path.dirname(__file__), 'lexicons/*.json')):
            with open(f_name, 'r') as json_f:
                lexicons[Path(f_name).stem] = load(json_f)
        return lexicons

    def template(self, template, words = None, iterations = 1):
        """
        Fill in a template. The template contains "[FORMAT]", "[MASK]", or one of the lexicon tokens.
        [FORMAT] - list of words to fill in the template must by provided in the `words` argument.
        The words are put into the correct word form.
        [MASK] - words to fill in the template are suggested using the MLM. Number of suggestions 
        can be specified using the `iterations` argument.
        [LEXICON] - random words are chosen from the "LEXICON" lexicon (if available) and filled 
        into the template in a correct form. Number of suggestions can again be specified using the 
        `iterations` argument.
        
        Parameters:
            template (str): str with one "[FORMAT]", "[MASK]", or [LEXICON] token
            words (List(str)): list of words to be filled in the template
            iterations (int): number of suggestions to return
        
        Returns:
            List(str): templates with the [TOKEN] replaced by word in correct form of length `iterations`.
        """

        tokens = template.split()
        tokens_set = set(tokens)
        word_forms = []
        mask_index = None

        tokens_in_template = []

        for spec_token in self.spec_tokens_mapping:
            if spec_token in tokens_set: tokens_in_template.append(spec_token)

        assert len(tokens_in_template) == 1, "Only one type of [SPECIAL] tokens is allowed in the sentence."
        spec_token = tokens_in_template[0]
        assert template.count(spec_token) == 1, "Only one [SPECIAL] token is allowed in the sentence."
        mask_index = tokens.index(spec_token)
        lexicon_type = self.spec_tokens_mapping[spec_token]

        # Correct word form requested
        if spec_token == "[FORMAT]":
            assert isinstance(words, list), "Words must be a list of strings."
            mask_index = tokens.index("[FORMAT]")
            tokens[mask_index] = "[MASK]"
            template = " ".join(tokens)
            for w in words:
                word_forms.extend(self._get_correct_word_form(w, template))

        # Language model suggestions requested
        elif spec_token == "[MASK]":
            assert iterations >= 1, "Iterations must be >= 1."
            mask_index = tokens.index("[MASK]")
            words = None
            template = template.strip()
            word_forms = self._generate_suggestions(template, suggestions_count = iterations)

        # Lexicon lookup requested
        else:
            assert iterations >= 1, "Iterations must be >= 1."
            tokens[mask_index] = "[MASK]"
            template = " ".join(tokens)
            random_words = np.random.choice(self.lexicons[lexicon_type], iterations)
            for w in random_words:
                word_forms.extend(self._get_correct_word_form(w, template))
        
        filled_sentences = []
        # Fill in the template
        assert mask_index is not None, "mask_index is None."
        for form in word_forms:
            if form is None:
                continue
            tokens[mask_index] = form
            sentence = " ".join(tokens)
            filled_sentences.append(sentence)
        return filled_sentences

    def _generate_suggestions(self, sentence, suggestions_count = 50):
        """
        Generate suggestions for a given sentence using the masked language model.
        
        Parameters:
            sentence (str): sentence containing [MASK] token
            suggestions_count (int): number of suggestions to generate

        Returns:
            List(str): list of suggestions
        """
        assert suggestions_count >= 1, "suggestions_count must be greater than 1"
        assert sentence.count("[MASK]") == 1, "Only one [MASK] token is allowed in the sentence."

        # Use language model to predict masked words
        inputs = self.tokenizer(sentence, return_tensors="np")
        token_logits = self.model(**inputs).logits

        # Find the location of [MASK] and extract its logits
        mask_token_index = np.argwhere(inputs["input_ids"] == self.tokenizer.mask_token_id)[0, 1]
        mask_token_logits = token_logits[0, mask_token_index, :]

        # Return the [MASK] candidates with the highest logits
        # We negate the array before argsort to get the largest, not the smallest, logits
        top = np.argsort(-mask_token_logits)[:suggestions_count].tolist()
        return [self.tokenizer.decode([result]).strip() for result in top]
        
    def _get_correct_word_form(self, word, context):
        """
        Determine the correct form of the word from the context. 
        Returns None if the word cannot be found in the context or MorphoDita cannot generate possible forms.

        Parameters:
            word (str): word to be corrected
            context (str): context in which the word is used

        Returns:
            List(str): list of correct forms of the word
        """
        assert word != None, "word cannot be None"
        assert context.count("[MASK]") == 1, "Only one [MASK] token is allowed in the sentence."

        # Get all the possible forms of the word
        generated = self.morphodita.generate(word)[0]
        if not generated: return list()

        tag_words = defaultdict(list)
        for gen in generated:
            tag_words[gen["tag"]].append(gen["form"])

        # Generate suggestions for the context
        top_tokens = self._generate_suggestions(context, suggestions_count = 50)

        tokens = context.split()
        mask_index = tokens.index("[MASK]")
        sentences = []
        for result in top_tokens:
            tokens[mask_index] = result
            sentences.append(" ".join(tokens))
        sentences_joined = "\n".join(sentences)

        # Get the possible tags from predicted words
        tagged = self.morphodita.tag(sentences_joined)
        tags = []
        for context in tagged:
            tag = context[mask_index]["tag"]
            if tag in tag_words:
                tags.append(tag)
        
        # Word cannot be found in the given context
        if not tags: return list()

        # Find the most frequent tag in a tag list
        most_freq_tag = max(set(tags), key = tags.count)
        return tag_words[most_freq_tag]

if __name__ == "__main__":
    e = Editor()
    print(e.list_special_tokens())
    print(e.template(" V [FORMAT] je krásně .", ["příroda", "město", "les", "Praha"]))
    for _ in range(1):
        print(e.template(" Bez [ZENA_JMENO] by se nám to dnes nepodařilo .", iterations=3))
        print(e.template(" [MUZ_JMENO] se má výborně ."))
        print(e.template(" Bydlí v [MESTO] ."))
        print(e.template(" Přestěhoval se sem v [MESIC] ."))
        print(e.template(" Bylo zrovna [DEN] ."))
        print(e.template(" Mám rád [MASK] .", iterations=3))
