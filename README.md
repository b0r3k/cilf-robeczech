# CILFuR
*Conjugated/Inflected Lemma Filling using Robeczech*

The tool allows for creating Czech sentences using *templates*. This does not sound that complicated, but Czech is a highly inflectional lagnuage and therefore it is always necessary to choose a correct word form. Templates can be filled in multiple ways – *providing a list of words* to be filled in, *specifying a predefined lexicon* to take the words from, or *letting the Masked Language Model provide suggestions*.

## Acknowledgements

The tool extensively uses RobeCzech (downloaded from [HuggingFace models](https://huggingface.co/ufal/robeczech-base)):
- Milan Straka, Jakub Náplava, Jana Straková and David Samuel: Czech RoBERTa, a monolingual contextualized language representation model. Accepted to TSD 2021.

and MorphoDiTa (its [web API hosted on Lindat](https://lindat.mff.cuni.cz/services/morphodita/api-reference.php)):
- (Straková et al. 2014) Straková Jana, Straka Milan and Hajič Jan. Open-Source Tools for Morphology, Lemmatization, POS Tagging and Named Entity Recognition. In Proceedings of 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pages 13-18, Baltimore, Maryland, June 2014. Association for Computational Linguistics. 

## Installation

For normal use, install `requirements.txt` (usually in a `venv`).

For development, install `requirements-dev.txt`. Installation using `python3 setup.py install` may also be necessary for the more complicated python imports to work.

## Usage

### Filling a template with provided word(s) in a correct form

### Using lexicons to fill in specific type of word in a correct form

### Letting the model suggest word(s) that fit the context

## How it works

Multiple words that fit the context are always generated using the MLM. If only the suggestions are required by the user, these are returned. In other cases, these suggestions are filled in into the sentence and it is morphologically tagged by MorphoDiTa (context aware tagger). Tags of the suggestions are then retrieved. The template is then filled in using words (specified by the user or randomly chosen from the predefined lexiocns) in forms specified by the tags retrieved. These forms are again generated using MorphoDiTa.

The lexicons were obtained using the code in [`/cilf_robeczech/lexicons/retrieval/`](/cilf_robeczech/lexicons/retrieval/), specifically:
- Names and surnames from some old Czech statistics (more details in the code), split into male and female using heuristics (female names end with `a`, `e` or `y`; female surnames end with `ová` or `í`) and hand-filtered afterwards.
- Colours, names of the cities and months were scraped from wikipedia; only the one word long were used due to the limitations of the tool.
- The 7 names of the days in a week were written by hand.

## Shortcomings – the future?

Multiple words/masks in a sentence support. Wordnet. Running MorphoDiTa locally.
