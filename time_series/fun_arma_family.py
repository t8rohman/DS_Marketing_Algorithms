'''
This function contains some algorithms to make a forecast of time series, 
and how to optimize the parameter using Grid Search method. 

The algorithms including:
--------------------------
optimize_ARIMA      : look for the best parameters for ARIMA
optimize_SARIMA     : look for the best parameters for SARIMA
optimize_SARIMAX    : look for the best parameters for SARIMAX
optimize_VARMAX     : look for the best parameters for VARMAX
rolling_forecast    : doing a rolling forecast, helpful for evaluating model from trained data
'''

import pandas as pd
import numpy as np
from typing import Union
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX


def optimize_ARIMA(endog: Union[pd.Series, list], order_list: list, d: int) -> pd.DataFrame:
    '''
    grid search function to look for the best parameters 
    for arima (p,d,q) with the lowest AIC
    
    Parameters
    ----------
    endog       : enter the time series value here
    order_list  : input all possible p, d, q you want to look for
    d           : how many differencing you did for the time series to become stationary
    
    Return
    ----------
    dataframe, showing all p,q values, sorted by the lowest AIC
    '''
    
    results = []
    
    # looping orders in order_list, wrapped in tqdm to monitor the progress
    # looping order to loop all the possible p and q that we want to check, and fit to SARIMAX by statsmodels
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(endog, 
                            order=(order[0], d, order[1]),
                            simple_differencing=False).fit(disp=False)
        except:
            continue
        
        # calculate the model's aic for every looping, then append it to a list
        aic = model.aic
        results.append([order, aic])
    
    # convert all the aic in the results list to a dataframe, then rename the columns
    result_df = pd.DataFrame(results)
    result_df.columns = ['p, q', 'AIC']
    
    # sorting it by the smallest AIC value
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


def optimize_SARIMA(endog: Union[pd.Series, list], 
                    order_list: list, d: int,
                    D: int, s: int) -> pd.DataFrame:
    '''
    grid search function to look for the best parameters 
    for sarima (p,d,q), (P,D,Q), and m with the lowest AIC
    
    Parameters
    ----------
    endog       : enter the time series value here
    order_list  : input all possible p, d, q you want to look for
    d           : how many differencing you did for the time series to become stationary
    D           : how many differencing for the seasonality, the same with d
    s           : number of time series in the season cycle
    
    Return
    ----------
    dataframe, showing all p,q,P,Q values, sorted by the lowest AIC
    '''
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try:
            # the only difference from ARIMA is when trying to fit the data
            # we add new parameter, "seasonal_order"
            model = SARIMAX(
                endog,
                order=(order[0], d, order[1]),      
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False).fit(disp=False)
        except:
            continue
        
        aic = model.aic
        results.append([order, aic])
        
    results_df = pd.DataFrame(results)
    results_df.columns = ['(p, q, P, Q', 'AIC']
        
    results_df = results_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
        
    return results_df 


def optimize_SARIMAX(endog: Union[pd.Series, list], 
                     exog: Union[pd.Series, list],  # the difference, defining exog variable
                     order_list: list,
                     d: int,
                     D: int,
                     s: int
                     ) -> pd.DataFrame:
    '''
    grid search function to look for the best parameters 
    for sarimax (p,d,q), (P,D,Q), and m with the lowest AIC
    plus exogenous variable, passed into 
    
    Parameters
    ----------
    endog       : enter the time series value here
    exog        : pass the exogenous variable to this variable
    order_list  : input all possible p, d, q you want to look for
    d           : how many differencing you did for the time series to become stationary
    D           : how many differencing for the seasonality, the same with d
    s           : number of time series in the season cycle
    
    Return
    ----------
    dataframe, showing all p,q,P,Q values, sorted by the lowest AIC
    '''
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(endog,
                            exog,
                            order=(order[0], d, order[1]),
                            seasonal_order=(order[2], D, order[3], s),
                            simple_differencing=False).fit(disp=False)
        except:
            continue
        
        aic = model.aic
        results.append([order, aic])
    
    result_df = pd.DataFrame(results)
    result_df.columns = ['p, q, P, Q', 'AIC']
    
    # sorting it by the smallest AIC value
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


def optimize_VARMAX(endog: Union[pd.Series, list],   # difference is here, will pas 2 variables
                    exog: Union[pd.Series, list],    # pass the exogenous variables here
                    order_list: list,                # only pass p and q iteration, without d, check the how-to later
                    ) -> pd.DataFrame:
    '''
    grid search function to look for the best parameters 
    for VARMAX (p,q), with the lowest AIC
    plus exogenous variable, passed into 
    
    Parameters
    ----------
    endog       : enter the time series value here
    exog        : pass the exogenous variable to this variable
    order_list  : input all possible p, q you want to look for
    
    Return
    ----------
    dataframe, showing all p,q, values, sorted by the lowest AIC
    '''
    
    
    results = []
    
    for order in tqdm_notebook(order_list):
        try:
            model = VARMAX(endog,
                           exog,
                           order=(order[0], order[1]),  # again, no d here, different with SARIMAX function
                           simple_differencing=False).fit(disp=False)
        except:
            continue
        
        aic = model.aic
        results.append([order, aic])
    
    result_df = pd.DataFrame(results)
    result_df.columns = ['p, q', 'AIC']
    
    # sorting it by the smallest AIC value
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


def rolling_forecast(endog,
                     exog,
                     data_len: int,
                     horizon: int,
                     window: int,
                     method: str,
                     kwargs_sarimax = {},   # to pass multiple kwargs, sarimax, https://stackoverflow.com/questions/26534134/python-pass-different-kwargs-to-multiple-functions
                     kwargs_varmax = {},    # to pass multiple kwargs, varmax
                     ) -> list:                  
    '''
    make a rolling forecast for time series data after splitting the data by train and test 
    available method using naive forecast, sarimax, and varmax. varmax only available to forecast
    2 variables at the same time
    
    Parameters
    ----------
    endog       : enter the time series value here
    exog        : pass the exogenous variable to this variable
    data_len    : data length to feed the model, pass the length of the train data set
    horizon     : how many time ahead we want to predict
    window      : steps for making the prediction, usually passing 1
    method      : method for forecasting, including mean, last value, and sarimax
    
    **kwargs_sarimax
        pass the parameter for statsmodels.tsa.statespace.sarimax.SARIMAX
        usually, passing the order and seasonal_order for SARIMAX
    **kwargs_varmax
        pass the parameter for statsmodels.tsa.statespace.varmax.VARMAX
        usually, passing the order of Auto Regressive and Moving Average Model
        
    Return
    ----------
    dataframe, showing all p,q,P,Q values, sorted by the lowest AIC
    '''
    
    total_len = data_len + horizon
    pred_value = []
    pred_value_i = []
    
    # naive forecast based on last value, as a comparison
    
    if method == 'last':

        for i in range(data_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            last_value_i = endog.index[i: i + window]
            
            pred_value.extend(last_value for _ in range(window))
            pred_value_i.extend(last_value_i)
        
        print(len(pred_value[:horizon]))
        print(len(pred_value_i))
    
        return pd.Series(pred_value[:horizon], index=pred_value_i)
    
    # naive forecast based on the mean values
    
    elif method == 'mean':
        
        for i in range(data_len, total_len, window):
            mean_value = np.mean(endog[:i].values)
            last_value_i = endog.index[i: i + window]
            pred_value.extend(mean_value for _ in range(window))
            pred_value_i.extend(last_value_i for _ in range(window))
            
        return pd.Series(pred_value[:horizon], index=pred_value_i)
    
    # forecast based on sarimax model
    
    elif method == 'sarimax':
        
        for i in range(data_len, total_len, window):
            model = SARIMAX(endog[:i],
                            exog[:i],
                            simple_differencing=False,
                            **kwargs_sarimax)
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog[:i])             # to get in- and out-of-sample prediction
            
            oos_pred = predictions.predicted_mean.iloc[-window:]        # get the out of sample prediction from the last (n) sample 
            oos_pred_i = endog.index[i:i + window]                      # get the out of sample index
            
            pred_value.extend(oos_pred)
            pred_value_i.extend(oos_pred_i)
    
        return pd.Series(pred_value[:horizon], index=pred_value_i)
    
    # forecast based on varmax model
    
    elif method == 'varmax':
        
        pred_value_1 = []
        pred_value_2 = []

        total_len = data_len + horizon

        for i in range(data_len, total_len, window):
            model = VARMAX(endog[:i],
                           exog[:i],
                           simple_differencing=False,
                           **kwargs_varmax
                           )
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog[:i])

            oos_pred_1 = predictions.predicted_mean.iloc[-window:, 0]
            oos_pred_2 = predictions.predicted_mean.iloc[-window:, 1]

            pred_value_1.extend(oos_pred_1)
            pred_value_2.extend(oos_pred_2)
            
            oos_pred_i = endog.index[i: i + window]
            pred_value_i.extend(oos_pred_i)
        
        return pd.Series(pred_value_1[:horizon], index=pred_value_i), pd.Series(pred_value_2[:horizon], index=pred_value_i)