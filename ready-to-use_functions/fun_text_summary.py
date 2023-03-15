'''
Collection of function to make a summary from a text based on several algorithms, 
including: TF-IDF, Latent Semantic Architecture (LSA), and Page Rank Summarizer.
'''


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import tokenize
import numpy as np

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer


def tfidf_summary(text, num_sum_sent):
    '''
    Make a summary based on the sum of TF-IDF weight.
    The most simple methodology to make a summary from a text.
    No need to specify the language first.
    
    text            : text that we want to make the summary of
    num_sum_sent    : number of sentence in the summary we want to have
    '''
    sum_list = []
    
    sentences = tokenize.sent_tokenize(text)
    tfidfVectorizer = TfidfVectorizer()
    words_tfidf = tfidfVectorizer.fit_transform(sentences)
    
    sent_sum = words_tfidf.sum(axis=1)
    important_sent = np.argsort(sent_sum, axis=0)[::-1]
    
    for i in range(len(sentences)):
        if i in important_sent[:num_sum_sent]:
            sum_list.append(sentences[i])
            
    return sum_list


def lsa_summary(text, num_sum_sent, lang):
    '''
    Make a summary based on Latent Semantic Architecure algorithm.
    Utilizing Sumy library to conduct the analysis.
    
    text            : text that we want to make the summary of
    num_sum_sent    : number of sentence in the summary we want to have
    lang            : language of the text, please check the sumy documentation for the available language
    '''
    sum_list = []
    lang = lang
    stemmer = Stemmer(lang)

    parser = PlaintextParser.from_string(text, Tokenizer(lang))
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(lang)
    
    for sentence in summarizer(parser.document, num_sum_sent):
        sum_list.append(str(sentence))
    
    return sum_list


def pagerank_summary(text, num_sum_sent, lang):
    '''
    Make a summary based on Page Rank algorithm.
    Inspired from the first Google Algorithm on ranking the pages.
    Utilizing Sumy library to conduct the analysis.
    
    text            : text that we want to make the summary of
    num_sum_sent    : number of sentence in the summary we want to have
    lang            : language of the text, please check the sumy documentation for the available language
    '''
    
    sum_list = []
    lang = lang
    stemmer = Stemmer(lang)
    
    parser = PlaintextParser.from_string(text, Tokenizer(lang))
    summarizer = TextRankSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(lang)
    
    for sentence in summarizer(parser.document, num_sum_sent):
        sum_list.append(str(sentence))
    
    return sum_list