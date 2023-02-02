'''
Set of functions to do a simple text preprocessing on a corpus of document that we have.
Functions referred to Blueprints for Text Analytics by Albrecht et al. (2021) with several adjustments to make it more clear. 
For the tutorial on how to use this, please refer to the notebook activity-1_simple-preprocessing.ipynb
'''

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

import regex as re  # regular expression
from textacy.extract.kwic import keyword_in_context as KWIC  # to make keyword-in-context (KWIC) analysis
from wordcloud import WordCloud  # to make word clouds

from collections import Counter  # to count list contains, similar to value_counts(), but faster


def tokenize(text):
    '''
    tokenize() function to tokenize text
    
    Parameters:
    text (str)    : pass the text you want to tokenize here
    
    Returns:
    text that has been tokenized
    '''
    
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)


def remove_stop(tokens, stopwords):
    '''
    remove_stop() function to remove stop words from a token
    
    Parameters:
    tokens          : pass the text that has been tokenized
    stopwords (set) : insert the stopwords, should be in 'set' type
    
    Returns:
    tokenized text with removed stopwords
    '''
    
    return [t for t in tokens if t.lower() not in stopwords]


def prepare(text, pipeline):
    '''
    prepare() function to do every preprocessing steps for the text we define
    
    Parameters:
    text (series)   : pass the text we want to tokenize
    pipeline        : pass one/all of these functions [str.lower, tokenize, remove_stop]
    
    Returns:
    all the preprocessing you define first, returned in series format
    '''
    
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens


def count_words(df, column='tokens', preprocess=None, min_freq=2):
    '''
    count_words() is a function to count how many words appear in the text
    
    Parameters:
    df          : the data frame containing the corpus
    columns     : column that contains token (should be tokenized first into list)
    preprocess  : process token and update counter
    min_freq    : min_frequency for a word in the text
    
    Returns:
    all the preprocessing you define first, returned in series format
    '''
    
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)
    
    counter = Counter()
    df[column].map(counter.update)
    
    df_freq = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    df_freq = df_freq[df_freq['freq'] > min_freq]
    df_freq.index.name = 'token'
    
    return df_freq.sort_values('freq', ascending=False)


def wordcloud_freq(word_freq, title, max_words=200, stopwords=None):
    '''
    wordcloud_freq() is a function to make a wordcloud based on the frequency dataframe
    
    Parameters:
    word_freq   : a series that contains the token and its frequency, should be a series
    title       : the title of wordcloud that you want to set up
    max_words   : how many words in maximum you want to plot
    stopwords   : set that contain stop words
    
    Returns:
    showing the plot in form of wordcloud, can be subplotted if you want
    '''
    
    # define the wc method first
    wc = WordCloud(width=800, height=400,
                   background_color='black', colormap='Paired',
                   max_font_size=150, max_words=max_words)
    
    # convert dataframe into dictionairy
    counter = word_freq.fillna(0).to_dict()
    
    # filter stop words in frequency counter
    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items()
                   if token not in stopwords}
        
    wc.generate_from_frequencies(counter)  # generate wc from a dictionary that contain frequency that words appear
    
    plt.title(title)
    
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    

def compute_idf(df, column, preprocess=None, min_freq=2):
    '''
    compute_idf() is a function to count how many words appear in the text, then inversed (tf-idf)
    
    Parameters:
    df                  : the data frame
    columns (series)    : column that contains token (should be tokenized first into list)
    preprocess          : process token and update counter
    min_freq            : min_frequency for a word in the text
    
    Returns:
    Dataframe with frequency of the words, and the idf weight of it.
    If we want to calculate tf-idf, we still have to multiply the tf by idf (tf x idf)
    '''

    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))  # each token is counted only once per document
    
    counter = Counter()
    df[column].map(update)
    
    df_idf = pd.DataFrame.from_dict(counter, orient='index', columns=['doc_freq'])  # caculate the frequency of word in every document first
    df_idf = df_idf[df_idf['doc_freq'] > min_freq]
    df_idf['idf'] = np.log(len(df)/(df_idf['doc_freq']+1))  # calculate idf, the inverse of document frequency
    df_idf.index.name = 'token'

    return df_idf.sort_values('doc_freq', ascending=False)


def kwic(doc_series, keyword, window=35, print_samples=5):
    '''
    kwic() is a function to do the keyword-in-context (KWIC) analysis
    
    Parameters:
    doc_series      : the series containing the text
    keyword         : word that we want to analyze, as the keyword
    windows         : how many words to the left and to the right
    print_samples   : how many samples do you want to see
    
    Returns:
    Text with the analysis of KWIC
    '''
    
    # make the list of kwic 
    # using the KWIC function from textacy library
    
    def add_kwic(text):
        kwic_list.extend(KWIC(text, keyword, ignore_case=True,
                              window_width=window))
    kwic_list = []
    doc_series.map(add_kwic)
    
    if print_samples is None or print_samples == 0:
        return kwic_list
    else:
        k = min(print_samples, len(kwic_list))
        
        print(f'{k} random samples out of {len(kwic_list)} contexts {keyword}:')
        
        # to print kwic list
        # re.sub is used to replace \n string to space as there's a lot of \n in the text
        # IMPORTANT! Change this one based on needs
        
        i = 1
        for sample in random.sample(kwic_list, k):
            print(str(i) + ') ' + (re.sub(r'[\n\t]', ' ', sample[0])) + ' ' + \
                sample[1] + ' ' + \
                    (re.sub(r'[\n\t]', ' ', sample[2])))
            i += 1
            

def ngrams(tokens, n=2, sep=' ', stopwords=set()):
    '''
    ngrams() frequency analysis for phrases (combination between 2 or 3 words)
    
    Parameters:
    tokens      : a list of words that has been tokenized
    n           : number of n grams (2 or 3, set default at 2)
    sep         : set the separator between words
    stopwords   : set the stopwords
    
    Returns:
    Series of text that has been analyzed using ngrams
    '''

    return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])
            if len([t for t in ngram if t in stopwords]) == 0]
    

def count_keywords_list(tokens, keywords):
    '''
    count_keywords_list() count words appear in a list by referring it to another list
    
    Parameters:
    tokens (list)      : a list of words from a document that has been tokenized
    keywords (list)    : list of keywords we want to count
    
    Returns:
    List of count keywords in the list/document
    '''

    tokens = [t for t in tokens if t in keywords]  # only calculate the appearence of token listed in keywords
    counter = Counter(tokens)  # count the appearence of the token in the keyword
    
    return [counter.get(k, 0) for k in keywords]  
    

def count_keywords_corpus(df, by, keywords, column='tokens'):
    
    '''
    count_keywords_corpus() frequency analysis for phrases (combination between 2 or 3 words)
    
    Parameters:
    df                  : the main dataframe of the corpus
    by                  : grouping by which column
    keywords (list)     : list of words that we want to check
    column (series)     : column that contain the tokens
    
    Returns:
    Dataframe of the words appearence count
    '''
    
    freq_matrix = df[column].apply(count_keywords_list, keywords=keywords)  # utilize count_keywords_list
    freq_df = pd.DataFrame.from_records(freq_matrix, columns=keywords)
    freq_df[by] = df[by]  # copy the column
    
    return freq_df.groupby(by).sum().sort_values(by)