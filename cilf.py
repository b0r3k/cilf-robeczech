from transformers import TFAutoModelForMaskedLM, AutoTokenizer
import numpy as np
from collections import defaultdict
from morph_api import MorphoDiTa

model_checkpoint = "ufal/robeczech-base"
model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


while True:
    text = input("Text:\t")
    word = input("Word:\t")
    #pos = input("POS:\t")
    
    #pos = "A"
    #text = "Tohle je [MASK] hračka než hrnec ."
    #text = "Tohle je [MASK] hračka ."
    inputs = tokenizer(text, return_tensors="np")
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = np.argwhere(inputs["input_ids"] == tokenizer.mask_token_id)[0, 1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    # We negate the array before argsort to get the largest, not the smallest, logits
    top_10_tokens = np.argsort(-mask_token_logits)[:50].tolist()

    tokens = text.split()
    mask_index = tokens.index("[MASK]")
    print(mask_index)
    sentences = []
    for result in top_10_tokens:
        tokens[mask_index] = tokenizer.decode([result]).strip()
        sentences.append(" ".join(tokens))
    sentences_joined = "\n".join(sentences)

    mt = MorphoDiTa()
    generated = mt.generate(word)[0]
    tag_words = defaultdict(list)
    for gen in generated:
        tag_words[gen["tag"]].append(gen["form"])

    tagged = mt.tag(sentences_joined)
    tags = []
    for sentence in tagged:
        tag = sentence[mask_index]["tag"]
        if tag in tag_words:
            tags.append(tag)
            print(tag, "\t", sentence[mask_index]["token"])
    
    # Find the most frequent tag in a tag list
    most_freq_tag = max(set(tags), key = tags.count)
    print(tag_words[most_freq_tag])
    # results = set()
    # for tag in tags:
    #     results.update(tag_words[tag])
    
    # print(results)

"""
POS:
A - adjektivum (přídavné jméno) 
C - numerál (číslovka, nebo číselný výraz s číslicemi) 
D - adverbium (příslovce) 
I - interjekce (citoslovce) 
J - konjunkce (spojka) 
N - substantivum (podstatné jméno) 
P - pronomen (zájmeno) 
R - prepozice (předložka) 
T - partikule (částice) 
V - verbum (sloveso) 
X - neznámý, neurčený, neurčitelný slovní druh 
Z - interpunkce, hranice věty
"""