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

    def template(self, template, words = None, iterations = 1):
        """
        TODO: Add description
        Parameters:
            template (str): str with one [MASK] token
            words (List(str)): list of word to be filled in the template
            iterations (int): number of iterations to perform
        
        Returns:
            List(str): templats with [MASK] replaced by word in correct form iteration-times
        """
        tokens = template.split()
        tokens_set = set(tokens)
        word_forms = []

        # Correct word form
        if "[FORMAT]" in tokens_set:
            mask_index = tokens.index("[FORMAT]")
            tokens[mask_index] = "[MASK]"
            template = " ".join(tokens)
            for w in words:
                word_forms.append(self._get_correct_word_form(w, template))
            # word_forms = self._get_correct_word_form(words, template)
        # Language model suggestions
        elif "[MASK]" in tokens_set:
            words = None
            template = template.strip()
            word_forms = self._generate_suggestions(template, suggestions_count = iterations)
        # Lexicon lookup
        else:
            lexicon_type = ""
            words = None
            if "[DEN]" in tokens_set:
                lexicon_type = "days"
                mask_index = tokens.index("[DEN]")
            elif "[MĚSÍC]" in tokens_set:
                lexicon_type = "months"
                mask_index = tokens.index("[MĚSÍC]")
            elif "[MUŽSKÉ_JMÉNO]" in tokens_set:
                lexicon_type = "male_names"
                mask_index = tokens.index("[MUŽSKÉ_JMÉNO]")
            elif "[ŽENSKÉ_JMÉNO]" in tokens_set:
                lexicon_type = "female_names"
                mask_index = tokens.index("[ŽENSKÉ_JMÉNO]")
            elif "[MĚSTO]" in tokens_set:
                lexicon_type = "cities"
                mask_index = tokens.index("[MĚSTO]")
            elif "[BARVA]" in tokens_set:
                lexicon_type = "colors"
                mask_index = tokens.index("[BARVA]")
            elif "[MUŽSKÉ_PŘÍJMENÍ]" in tokens_set:
                lexicon_type = "male_surnames"
                mask_index = tokens.index("[MUŽSKÉ_PŘÍJMENÍ]")
            elif "[ŽENSKÉ_PŘÍJMENÍ]" in tokens_set:
                lexicon_type = "female_surnames"
                mask_index = tokens.index("[ŽENSKÉ_PŘÍJMENÍ]")
            else:
                print("Unknown mask type.")
                return

            tokens[mask_index] = "[MASK]"
            template = " ".join(tokens)
            random_words = np.random.choice(self.lexicons[lexicon_type], iterations)
            for w in random_words:
                word_forms.append(self._get_correct_word_form(w, template))
            # word_forms = self._get_correct_word_form(random_word, template)
        
        # if word_forms == None:
        #     if words != None:
        #         print(f"Word {words} cannot be found in given context.")
        #     return
        
        for form in word_forms:
            tokens[mask_index] = form
            print(" ".join(tokens))

    def _generate_suggestions(self, sentence, suggestions_count = 50):
        """
        Generate suggestions for a given sentence using the language model.
        
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
        if generated == []:
            return None

        tag_words = defaultdict(list)
        for gen in generated:
            tag_words[gen["tag"]].append(gen["form"])

        # Generate suggestions for the context
        top_tokens = self._generate_suggestions(context, suggestions_count = 50)

        tokens = context.split()
        mask_index = tokens.index("[MASK]")
        sentences = []
        for result in top_tokens:
            # tokens[mask_index] = self.tokenizer.decode([result]).strip()
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
        if not tags:
            return None

        # Find the most frequent tag in a tag list
        most_freq_tag = max(set(tags), key = tags.count)
        return tag_words[most_freq_tag][0]

# while True:
#     # text = input("Text:\t")
#     # word = input("Word:\t")
    
#     model_checkpoint = "ufal/robeczech-base"
#     model = TFAutoModelForMaskedLM.from_pretrained
#     tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#     text = "Tohle je [MASK] hračka než [MASK] ."
#     inputs = tokenizer(text, return_tensors="np")
#     token_logits = model(**inputs).logits
#     outputs = model(**inputs)
#     predictions = outputs[0]
#     sorted_preds, sorted_idx = predictions[0].sort(dim=-1, descending=True)
#     for k in range(10):
#         predicted_index = [sorted_idx[i, k].item() for i in range(0,24)]
#         predicted_token = [tokenizer.convert_ids_to_tokens([predicted_index[x]])[0] for x in range(1,24)]
#         print(predicted_token)
#         break

#     text = "Mám rád [MASK] [MASK] ."
#     inputs = tokenizer(text, return_tensors="np")
#     token_logits = model(**inputs).logits
#     # Find the location of [MASK] and extract its logits
#     mask_token_indices = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[:, 1]
#     mask_token_logits = [token_logits[0, mask_token_index, :] for mask_token_index in mask_token_indices]
#     # Pick the [MASK] candidates with the highest logits
#     # We negate the array before argsort to get the largest, not the smallest, logits
#     for result0, result1 in zip(mask_token_logits[0], mask_token_logits[1]):
#         print(tokenizer.decode([result0]).strip(), tokenizer.decode([result1]).strip())
#     #top_10_tokens = [np.argsort(-mask_token_logit)[:50].tolist() for mask_token_logit in mask_token_logits]
#     #for result0, result1 in zip(top_10_tokens[0], top_10_tokens[1]):
#     #    print(tokenizer.decode([result0]).strip(), tokenizer.decode([result1]).strip())

#     # mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
#     # mask_token_logits = token_logits[0, mask_token_index, :]
#     # # Pick the [MASK] candidates with the highest logits
#     # # We negate the array before argsort to get the largest, not the smallest, logits
#     # top_10_tokens = np.argsort(-mask_token_logits)[:50].tolist()
#     # for result in top_10_tokens:
#     #     print(tokenizer.decode([result]).strip())


#     # from transformers import pipeline
#     # mask_filler = pipeline("fill-mask", model=model, tokenizer=tokenizer)
#     # mask_filler("Tohle je [MASK].")

if __name__ == "__main__":
    e = Editor()
    # print(e.lexicons.keys())
    e.template(" V [FORMAT] je krásně .", "příroda")
    for _ in range(1):
        e.template(" Bez [ŽENSKÉ_JMÉNO] by se nám to dnes nepodařilo .")
        e.template(" [MUŽSKÉ_JMÉNO] se má výborně .")
        e.template(" Bydlí v [MĚSTO] .")
        e.template(" Přestěhoval se sem v [MĚSÍC] .")
        e.template(" Mám rád [MASK] .")