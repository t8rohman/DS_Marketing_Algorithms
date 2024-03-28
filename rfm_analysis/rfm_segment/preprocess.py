import pandas as pd
import numpy as np
from datetime import datetime

from scipy import stats
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


class PreprocessTransaction:
    def __init__(self, df, id_col, rfm_cols):
        self.df = df
        self.id_col = id_col
        self.rfm_cols = rfm_cols


    def pivot_transacions(self):
        df = self.df
        id_col = self.id_col
        rfm_cols = self.rfm_cols

        df_rfm = df.groupby(id_col).agg({rfm_cols[0]: 'max',
                                         rfm_cols[1]: 'count',
                                         rfm_cols[2]: 'sum'})
        
        today = pd.Timestamp(datetime.now().date())

        df_rfm[rfm_cols[0]] = pd.to_datetime(df_rfm[rfm_cols[0]])
        df_rfm['last_purchase'] = (today - df_rfm[rfm_cols[0]]).dt.days

        df_rfm = df_rfm[['last_purchase', rfm_cols[1], rfm_cols[2]]]
        df_rfm = df_rfm.rename(columns={'last_purchase': 'Recency',
                                        rfm_cols[1]: 'Frequency',
                                        rfm_cols[2]: 'Monetary'})
        
        return df_rfm


class PreprocessKMeans:
    '''
    A class for preprocessing data including normalization and skewness checking.

    Attributes:
        df (DataFrame): The input DataFrame.
        df_processed (DataFrame): The processed DataFrame.
    '''

    def __init__(self, df):
        '''
        Initialize the PreprocessData object.

        Args:
            df (DataFrame): The input DataFrame to be processed.
        '''
        self.df = df
        self.df_processed = None

    def normalize_log(self):
        '''
        Apply log transformation to the DataFrame.

        Returns:
            DataFrame: The DataFrame with log-transformed values.
        '''
        df = self.df
        df = np.log(df + 1)
        self.df_processed = df
        return df

    def normalize_boxcox(self):
        '''
        Apply Box-Cox transformation to the DataFrame.

        Returns:
            DataFrame: The DataFrame with Box-Cox transformed values.
        '''
        df = self.df
        for col in df.columns:
            transformed_data, _ = stats.boxcox(df[col])
            df[col] = transformed_data
        self.df_processed = df
        return df

    @staticmethod
    def check_skew(df, col):
        '''
        Check the skewness of a column in the DataFrame and plot its distribution.

        Args:
            df (DataFrame): The DataFrame.
            col (str): The column name.

        Returns:
            None
        '''
        skew_val = stats.skew(df[col])
        skew_test = stats.skewtest(df[col])

        sns.displot(df[col], kde=True, height=4, aspect=1.5)
        plt.title('Distribution of ' + col)

        print("{}'s: Skew: {}, : {}".format(col, skew_val, skew_test))

        plt.show()

    def check_skew_all(self):
        '''
        Check the skewness of all columns in the DataFrame and plot their distributions.

        Returns:
            None
        '''
        df = self.df_processed if hasattr(self, 'df_processed') else self.df

        for col in df.columns:
            self.check_skew(df, col)


def standardize(df):
    scaler = StandardScaler()
    scaler.fit(df)

    df_rfm_scaled = scaler.transform(df)
    df_rfm_scaled = pd.DataFrame(df_rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    df_rfm_scaled.head()

    return df_rfm_scaled
