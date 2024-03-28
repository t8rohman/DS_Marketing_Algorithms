import pandas as pd
import numpy as np

class RFMTable:
    '''
    Class for generating RFM (Recency, Frequency, Monetary) scores and segments from a given DataFrame.

    Attributes:
    - df_rfm (DataFrame): The input DataFrame containing Recency, Frequency, and Monetary values.

    Methods:
    - rfm_table(): Calculate RFM scores and segments based on the input DataFrame and return the modified DataFrame.
    '''

    def __init__(self, df_rfm):
        '''
        Initialize the RFMTable object.

        Args:
            df_rfm (DataFrame): The input DataFrame containing Recency, Frequency, and Monetary values.
        '''
        self.df_rfm = df_rfm

    def rfm_table(self):
        '''
        Calculate RFM scores and segments based on the input DataFrame.

        Returns:
        DataFrame: The modified DataFrame with added columns for RFM scores and segments.
        '''
        df_rfm = self.df_rfm

        # save the values for every quintile we set
        # will be used later if we want to do the rfm segmentation with the same binning again
        q_rec = pd.qcut(df_rfm['Recency'], q=5, labels=[5,4,3,2,1], retbins=True)[1]
        q_freq = pd.qcut(df_rfm['Frequency'], q=5, labels=[5,4,3,2,1], retbins=True)[1]
        q_mon = pd.qcut(df_rfm['Monetary'], q=5, labels=[5,4,3,2,1], retbins=True)[1]

        # create rfm score columns
        df_rfm['r_score'] = pd.qcut(df_rfm['Recency'], q=5, labels=[5,4,3,2,1])
        df_rfm['f_score'] = pd.qcut(df_rfm['Frequency'], q=5, labels=[1,2,3,4,5])
        df_rfm['m_score'] = pd.qcut(df_rfm['Monetary'], q=5, labels=[1,2,3,4,5])

        # combine all rfm scores
        # later on to make the segmentation
        df_rfm['rfm_segment'] = df_rfm['r_score'].astype('str') + df_rfm['f_score'].astype('str') + df_rfm['m_score'].astype('str')
        df_rfm['rfm_score'] = df_rfm['r_score'].astype('int') + df_rfm['f_score'].astype('int') + df_rfm['m_score'].astype('int')

        seg_map = {
            r'[1-2][1-2]': 'Hibernating',
            r'[1-2][3-4]': 'At Risk',
            r'[1-2]5': 'Can\'t Loose',
            r'3[1-2]': 'About to Sleep',
            r'33': 'Need Attention',
            r'[3-4][4-5]': 'Loyal Customers',
            r'41': 'Promising',
            r'51': 'New Customers',
            r'[4-5][2-3]': 'Potential Loyalists',
            r'5[4-5]': 'Champions'
        }

        df_rfm['rfm_cat_seg'] = (df_rfm['r_score'].astype('str') + df_rfm['f_score'].astype('str')).replace(seg_map, regex=True)
        
        return df_rfm