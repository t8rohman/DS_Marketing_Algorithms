'''
This script contains all the functions needed to run an NLP preprocessing using spacy and textacy.
Process the data from the data cleaning steps, to the linguistic process.
Functions referred to Blueprints for Text Analytics by Albrecht et al. (2021) with several adjustments to make it more clear.
For the tutorial on how to use this, please refer to the notebook activity-2_nlp-spacy-preprocessing.ipynb
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import html
import spacy
import tqdm

import regex as re  # regular expression
import textacy
import textacy.preprocessing as tprep
from textacy.extract.kwic import keyword_in_context as KWIC  # to make keyword-in-context (KWIC) analysis

nlp = spacy.load('en_core_web_lg')  # setting the language


def impurity(text, min_len=10):
    ''''calculate the impurity of the text
    
    Parameters:
    text    : text that we want to analyze
    min_len : minimal length of the text, else it will return 0
    
    Returns
    impurity level of the text
    '''
    RE_SUSPICIOUS = re.compile(r'[&#<>{}\[\]\\]')  # adjust this based on pattern we want to calculate as impurity
    
    if text == None or len(text) < min_len:
        return 0
    else:
        return len(RE_SUSPICIOUS.findall(text))/len(text)
    

def clean(text):
    '''
    IMPORTANT! Make some adjustment for this function based on the text you have!
    
    Blueprint function first substitutes all HTML escapes
    by their plain-text representation and then replaces certain patterns by spaces. 
    Finally, sequences of whitespaces are pruned.
    
    Taken directly from the book.
    '''
    
    # convert html escapes like &amp; to characters.
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs like [Some text](https://....)
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text or code in brackets like [0]
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials, matches &# but not #cool
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text) # standalone sequences of hyphens like --- or ==
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of white spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize(text):
    '''
    IMPORTANT! Make some adjustment for this function based on the text you have!
    There's a lot of normalization functions from textacy we can use, see the documentation!
    
    Normalize text, the list of the functions are shown below
    '''
    
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text


def display_prop_spacy(doc, include_punct=False):
    '''
    function to show all the properties of the token in form of dataframe
    
    Parameters:
    doc             : the preprocessed text and already converted as spaCy object
                      means -> doc = nlp(text)
    include_punct   : True/False
    
    Return:
    dataframe of spacy properties
    '''
    
    rows = []
    for i, t in enumerate(doc):
        if not t.is_punct or include_punct:
            row = {'index': i, 'text': t.text, 'lemma_': t.lemma_,
                   'is_stop': t.is_stop, 'is_alpha': t.is_alpha,
                   'pos_': t.pos_, 'dep_': t.dep_,
                   'ent_type_': t.ent_type_, 'ent_iob': t.ent_iob_}
            rows.append(row)
    
    df = pd.DataFrame(rows).set_index('index')
    df.index.name = None
    
    return df


from spacy.tokenizer import Tokenizer
from spacy.util import compile_prefix_regex, compile_infix_regex, compile_suffix_regex


def custom_tokenizer_spacy(nlp):
    
    '''
    functions to modify already-made tokenizer rule from spacy
    be careful for any changes, even slight changes could distort the data a lot
    '''
    prefixes = [pattern for pattern in nlp.Defaults.prefixes if pattern not in ['-', '_', '#']]
    suffixes = [pattern for pattern in nlp.Defaults.suffixes if pattern not in ['_']]
    infixes = [pattern for pattern in nlp.Defaults.infixes if not re.search(pattern, 'xx-xx')]
    
    return Tokenizer(vocab = nlp.vocab,
                     rules = nlp.Defaults.tokenizer_exceptions,
                     prefix_search  = compile_prefix_regex(prefixes).search,
                     suffix_search  = compile_suffix_regex(suffixes).search,
                     infix_finditer = compile_infix_regex(infixes).finditer,
                     token_match    = nlp.Defaults.token_match)

nlp.tokenizer = custom_tokenizer_spacy(nlp)



# Collection of Extracting Words Based on their Part-of-Speech (POS) Tagging


def extract_lemmas(doc, **kwargs):
    '''to extract the lemmatized form of the text
    
    Parameters:
    doc         : preprocessed text using nlp
    **kwargs    : follow the parameters from textacy.extract.words
    '''
    return [t.lemma_ for t in textacy.extract.words(doc, **kwargs)]


def extract_noun_phrases(doc, pos_pattern, sep='_'):
    '''to extract noun-phrases based on pattern of POS
    
    Parameters:
    doc         : preprocessed text using nlp
    pos_pattern : list of the pattern for POS
    sep         : separator between words
    
    Return:
    clean text that has been preprocessed.
    '''
    patterns = []
    for pos in pos_pattern:
        '''
        change here if you want to make any changes for the patterns
        especially in POS:NOUN:+
        based on spaCy regex rules, check: https://spacy.io/usage/rule-based-matching
        '''
        patterns.append(f'POS:{pos} POS:NOUN:+')        
    spans = textacy.extract.matches.token_matches(doc, patterns=patterns)
    
    return [sep.join([t.lemma_ for t in s]) for s in spans]


def extract_entities(doc, include_types=None, sep='_'):
    '''to extract named entity based from
    
    Parameters:
    doc             : preprocessed text using nlp
    include_types   : list of the type of entities we want to see
    sep             : separator between words
    
    Return:
    clean text that has been preprocessed.
    '''
    ents = textacy.extract.entities(doc,
                                    include_types=include_types,
                                    exclude_types=None,
                                    drop_determiners=True,
                                    min_freq=1)
    
    return [sep.join([t.lemma_ for t in e]) + '/' + e.label_ for e in ents]



# Wrapper Function from all the Extraction Functions


def extract_nlp(doc):
    '''
    collection of the extraction functions we build before.
    wrapper of functions, all in one.
    
    IMPORTANT! Adjust based on what you want to see from the data.
    '''
    
    return{
        'lemmas'            : extract_lemmas(doc),
        'adjs_verbs'        : extract_lemmas(doc, include_pos = ['ADJ', 'VERB']),
        'nouns'             : extract_lemmas(doc, include_pos = ['NOUN', 'PROPN']),
        'noun_phrases'      : extract_noun_phrases(doc, ['NOUN']),
        'adj_noun_phrases'  : extract_noun_phrases(doc, ['ADJ']),
        'entities'          : extract_entities(doc, ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'LOC'])
    }


def extract_nlp_df(doc):
    '''
    to see what we get from extract_nlp function
    
    Parameters:
    doc         : preprocessed text using nlp
    
    Return:
    dataframe of extract_nlp result
    '''
    
    i = 0
    df_nlp = pd.DataFrame()

    for col, values in extract_nlp(doc).items():
        df_nlp.loc[i, 'linguistic_att'] = col
        df_nlp.loc[i, 'values'] = ', '.join(values)
        i += 1
    
    return df_nlp


def extract_lemmas(text):
    '''
    extract lemmatized version of the text
    
    Parameters:
    text        : text that we want to lemmatize
    
    Return:
    lemmatized version of the text
    '''
    
    doc = nlp(str(text))
    text = ' '.join([token.lemma_ for token in doc])
    return text


def extract_pos_to_take(text, pos_to_take=['NOUN', 'PROPN', 'ADJ', 'ADV', 'VERB']):
    '''
    extract specific part-of-speech (POS) from the text, after being lemmatized first
    defaul pos are noun, proposition, adjective, adverb, and verb 
    
    Parameters:
    text        : text that we want to take
    
    Return:
    extracted version of the text
    '''
    
    doc = nlp(str(text))
    text = ' '.join([token.lemma_ for token in doc if token.pos_ in pos_to_take])
    return text