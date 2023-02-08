'''
Collection functions to make a syntatic similarity analysis

look_for_text()         : Function to look for the most similar document in a corpus from a defined text
similar_doc_in_corpus() : Function to search for the most 2 similar documents in a corpus from a defined text
similar_doc_in_corpus() : Function to find for similar words from documents in corpus
'''

import pandas as pd
import numpy as np

from tqdm import tqdm

# libraries for the machine learning models
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity


def look_for_text(doc, text_to_look_for, top_n, **kwargs):
    '''
    Function to look for the most similar document in a corpus from a defined text
    
    Parameters:
    doc                 : collection of docs we want to analyze (from a corpus)
    text_to_look_for    : text that we're looking for
    dt                  : data frame that has been vectorized
    top_n               : n top similar text we're looking for
    **kwargs            : pass the parameters from TfidfVectorizer
    
    Returns:
    Dataframe of top similar text with cosine similarity method
    '''
    
    tfidf = TfidfVectorizer(**kwargs)
    dt = tfidf.fit_transform(doc)
    
    look_for = tfidf.transform([text_to_look_for])
    sim = cosine_similarity(look_for, dt)
    
    x = np.argsort(sim)[0][::-1][:top_n]
    top_df = pd.DataFrame(doc.iloc[x].values, x, columns=['document'])
    for i in x:
        top_df.loc[i, 'cosine_sim'] = sim[0][i]
    
    return top_df


def similar_doc_in_corpus(doc, batch=10000, max_sim=0.0, **kwargs):
    '''
    Function to look for the most 2 similar documents in a corpus from a defined text.
    Beware of this as it takes long to finish.
    
    Parameters:
    doc                 : collection of docs we want to analyze (from a corpus)
    **kwargs            : pass the parameters from TfidfVectorizer
    
    Returns:
    Two documents with the highest pairwise-similarity
    '''
    
    batch = batch
    max_sim = max_sim

    max_a = None
    max_b = None
    
    tfidf = TfidfVectorizer(**kwargs)
    dt = tfidf.fit_transform(doc)

    for a in tqdm(range(0, dt.shape[0], batch)):
    
        for b in range(0, a+batch, batch): 
            # print(a, b) -> should be eliminated, the book says to print(a,b)
            r = np.dot(dt[a:a+batch], np.transpose(dt[b:b+batch]))
        
            # eliminate identical vectors
            # by setting their similarity to np.nan which gets sorted out r[r > 0.9999] = np.nan
            r[r > 0.9999] = np.nan
            sim = r.max()
        
            if sim > max_sim:
                # argmax returns a single value which we have to
                # map to the two dimensions
                (max_a, max_b) = np.unravel_index(np.argmax(r), r.shape)  
            
                # adjust offsets in corpus (this is a submatrix)
                max_a += a
                max_b += b
                max_sim = sim
        
    list_sim = [doc[max_a], doc['headline_text'][max_b]]
        
    return print('\n'.join((list_sim)))


def similar_word(doc, top_n, min_appear, **kwargs):
    '''
    Function to find for similar words from documents in corpus
    
    Parameters:
    doc         : collection of docs we want to analyze (from a corpus)
    top_n       : n top similar text we're looking for
    min_n       : minimal appearence of the pair
    **kwargs    : pass the parameters from TfidfVectorizer
    
    Returns:
    Dataframe of top similar word from a corpus with it's cosine similarity
    '''
    
    tfidf_word = TfidfVectorizer(min_df=min_appear, **kwargs)  
    dt_word = tfidf_word.fit_transform(doc)

    r = cosine_similarity(dt_word.T, dt_word.T)  # this is the part where we transpose the data
    np.fill_diagonal(r, 0)
    
    voc = tfidf_word.get_feature_names_out()  # create the vocabulary
    size = r.shape[0]
    
    # create the data frame and its row iteration
    df_sim = pd.DataFrame()
    row = 0
    
    for i in np.argsort(r.flatten())[::-1][0:100]:
        # finding the pair
        a = int(i/size)
        b = i%size
        if a > b:  # to avoid repetitions (only show the pair once)
            df_sim.loc[row, 'word_1'] = voc[a]
            df_sim.loc[row, 'word_2'] = voc[b]
            df_sim.loc[row, 'sim'] = r[a][b]
            row += 1
    
    return df_sim.head(top_n)