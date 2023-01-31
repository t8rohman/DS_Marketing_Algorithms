import pandas as pd
import numpy as np
import itertools

from sklearn.metrics.pairwise import cosine_similarity

# User-Based Collaborative Filtering
# Consists of 3 functions: build_user_user_matrix(), top_similar_cust(), and items_to_recommend()
# Should be ran in order to make it works


'''
build_user_user_matrix() is a function to calculate cosine similarity for user-based collaborative filtering
df      : dataframe that cointain your data
index   : pass the customer identifier column here
columns : pass the item/product identifier column here
values  : pass the column that contain how many items the customer bought
'''

def build_user_user_matrix(df, index, columns, values):
    
    customer_item_matrix = df.pivot_table(
    index=index,
    columns=columns,
    values=values,
    aggfunc='sum'
    )
    
    customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
    
    user_user_sim_matrix = pd.DataFrame(
    cosine_similarity(customer_item_matrix)
    )

    user_user_sim_matrix.columns = customer_item_matrix.index
    user_user_sim_matrix.index = customer_item_matrix.index
    
    return customer_item_matrix, user_user_sim_matrix


'''
top_similar_cust() is a function to show top similar customers to the customer we pass for the variable
user_user_sim_matrix    : pass user_user_sim_matrix from build_user_user_matrix()
index                   : pass the customer identifier column here
cust_main               : customer identifier as the main subject
num_sim                 : how many top users you want to know
'''

def top_similar_cust(user_user_sim_matrix, index, cust_main, num_sim):
    df_top_similar_cust = pd.DataFrame(user_user_sim_matrix.loc[cust_main].sort_values(ascending=False).reset_index(drop=False).head(num_sim))
    df_top_similar_cust.columns = [index, 'CosineSim']
    return df_top_similar_cust


'''
items_to_recommend() is a function to show what are the items/products should we recommend based on the similarity from top_similar_cust()
customer_item_matrix    : pass customer_item_matrix from build_user_user_matrix()
cust_main               : customer identifier as the main subject
cust_comp               : customer to be shown the products should we recommend
'''

def items_to_recommend(customer_item_matrix, cust_main, cust_comp):

    items_bought_by_A = set(customer_item_matrix.loc[cust_main].iloc[customer_item_matrix.loc[cust_main].to_numpy().nonzero()].index)
    items_bought_by_B = set(customer_item_matrix.loc[cust_comp].iloc[customer_item_matrix.loc[cust_comp].to_numpy().nonzero()].index)

    items_to_recommend_to_B = items_bought_by_A - items_bought_by_B
    list(items_to_recommend_to_B)

    return list(items_to_recommend_to_B)



# Item-Based Collaborative Filtering
# Consists of 2 functions: build_item_item_matrix(), top_similar_item_fun(), and items_to_recommend()
# Should be ran in order to make it works


'''
build_item_item_matrix() is a function to calculate cosine similarity for item-based collaborative filtering
df      : dataframe containing your data
index   : pass the customer identifier here
columns : pass the item/product identifier here
values  : pass the column that contain how many items the customer bought
'''

def build_item_item_matrix(df, index, columns, values):
    
    customer_item_matrix = df.pivot_table(
    index=index,
    columns=columns,
    values=values,
    aggfunc='sum'
    )
    
    customer_item_matrix = customer_item_matrix.applymap(lambda x: 1 if x > 0 else 0)
    
    item_item_sim_matrix = pd.DataFrame(cosine_similarity(customer_item_matrix.T))

    item_item_sim_matrix.columns = customer_item_matrix.T.index
    item_item_sim_matrix.index = customer_item_matrix.T.index
    
    return customer_item_matrix, item_item_sim_matrix


'''
top_similar_item_fun() is a function to calculate cosine similarity for user-based collaborative filtering
item_item_sim_matrix    : pass item_item_sim_matrix from build_item_item_matrix()
columns                 : pass the item/product identifier here
item_id                 : pass the item that you want to identify as the main subject
num_sim                 : number of similar item you want to see
'''

def top_similar_item_fun(item_item_sim_matrix, columns, item_id, num_sim='all'):
    
    if num_sim == 'all':
        top_similar_item = item_item_sim_matrix.loc[item_id].sort_values(ascending=False)
    elif isinstance(num_sim, int) == True:
        top_similar_item = item_item_sim_matrix.loc[item_id].sort_values(ascending=False).head(num_sim)
        
    df_recommend_item = pd.DataFrame(top_similar_item).reset_index()
    df_recommend_item.columns = [columns, 'CosineSim']
    
    return df_recommend_item