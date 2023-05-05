'''
This script used to make a Time Series analysis using a deep learning model approach
and comparing it to the baseline models.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTMCell, RNN, Dense
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.optimizers import Adam 


class DataWindow():
    '''
    A class to represent a DataWindow, where it will allow us to format the 
    data appropriately to be fed to our time series deep learning models.

    Attributes
    ----------
    input_width : int
        Number of timesteps that are fed into the model to make predictions.
    label_width : int
        Number of timesteps in the predictions.
    shift : int
        Number of timesteps separating the input and the predictions.
    train_df : pd.DataFrame()
        Dataframe for training.
    val_df : pd.DataFrame()
        Dataframe for validation.
    test_df : pd.DataFrame()
        Dataframe for the test.
    label_columns: list
        Name of the column that we wish to predict, pass it into a list first.
    

    Methods
    -------
    split_to_inputs_labels(features):
        Split the window between inputs and labels.
    plot(model, plot_col, max_subplots):
        Plot the input data, prediction, and actual values.
    make_dataset(data):
        Format the data set into tensors to make a deep learning model later.
    '''
    
    
    # 1 Defining the initialization function of DataWindow
    
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,  # REMIND ME for this line
                 label_columns=None):
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        self.label_columns = label_columns  # name of the column to predict
        if label_columns is not None:                                                           
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}  # dict: name and index of the label column, for PLOTTING
            self.column_indices = {name: i for i, name in enumerate(train_df.columns)}  # dict: name and index of each column
        
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)  # return slice object that specfifies how to slice a sequence
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]  # assign indices to the inputs, for PLOTTING
        
        self.label_start = self.total_window_size - self.label_width  # index at which the label starts
        self.labels_slice = slice(self.label_start, None)  # same like input slice, but now for the labels
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]
        
        
    # 2 Split the window between inputs and labels
    
    def split_to_inputs_labels(self, features):
        '''
        Split the window between inputs and labels

        Attributes
        ----------
        features:
            pass the list of the features here.
        '''
        
        inputs = features[:, self.input_slice, :]  # slice the window to get the inputs using the input_slice defined in initialization
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:  # if we have more than one target, stack the labels using tf keras
            labels = tf.stack([labels[:,:,self.column_indices[name]] for name in self.label_columns], axis=-1)
        
        # set shape would be [batch, time, features]
        
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        
        return inputs, labels
    
    
    # 3 Plot the input data, prediction, and actual values
    
    def plot(self, plot_col, model=None, max_subplots=3):  # REMIND ME ABOUT TRAFFIC VOLUME LATER!!!
        '''
        Plot the input data, prediction, and actual values.

        Attributes
        ----------
        model :
            Pass the model.
        plot_col :
            The target column we want to predict (y-axis).
        max_subplots:
            how many subplots do we want to have, based on the matplotlib.plotly plot.
        '''
        
        inputs, labels = self.sample_batch
        
        plt.figure(figsize=(12,8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        
        for n in range(max_n):
            plt.subplot(3, 1, n+1)
            plt.ylabel(f'{plot_col} [scaled]')
            
            # plotting the input, appear as a countinuous blue line with dots (change 'marker' and 'c' parameter)
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)
            
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            
            if label_col_index is None:
                continue
            
            # plotting the actual values, appear as line of green squares (change 'marker' and 'c' parameter)
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', marker='s', label='Labels', c='green', s=64)
            
            # plotting the prediction, appear as red crosess (change 'marker' and 'c' parameter)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions', c='red', s=64)
            
            if n == 0:
                plt.legend()
            
            plt.xlabel('Time (h')
                            
            
    # 4 Format the data set into tensors to make a deep learning model later
        
    def make_dataset(self, data):
        '''
        Format the data set into tensors to make a deep learning model later.

        Attributes
        ----------
        data :
            Pass in the data, correspond to the training, validation, and test set.
        '''
        
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32
            )
            
        ds = ds.map(self.split_to_inputs_labels)
        return ds
    
    
    # 5 Define property to get the data (getter)
    # To access private variables directly in Python, 
    
    @property
    def train(self):
        return self.make_dataset(self.train_df)
    
    @property
    def val(self):
        return self.make_dataset(self.val_df)
    
    @property
    def test(self):
        return self.make_dataset(self.test_df)
    
    @property
    def sample_batch(self):
        result = getattr(self, '_sample_batch', None)
        if result is None:
            result = next(iter(self.train))
            self._sample_batch = result
        return result
    

class Baseline(Model):  
    
    # Notes about the Model in Baseline() arguments:
    # This will import the Model class from the tensorflow.keras.models module, which is a subclass of the Layer class. 
    # Then, the Baseline class can inherit from the Model class and use its functionality without any issue.
    
    '''
    Create a baseline class, where it uses the last observed value (input data point)
    and make it as the next timestep prediction, based on the shift of the data. 

    Attributes
    ----------
    label_index : list
        Specify a list of targets we want to predict.

    Methods
    -------
    call(inputs):
        Return the data as TensorFlow instance.
    '''
    
    def __init__(self, label_index=None):
        super().__init__()  # super() function that will make the child class inherit all the methods and properties from its parent
        self.label_index = label_index
    
    
    def call(self, inputs):
        if self.label_index is None:  # if no target is specified, then it will return all columns
            return inputs
        
        elif isinstance(self.label_index, list):  # if we specify a list of targets, it will return only the specified columns
            tensors = []
            for index in self.label_index:
                result = inputs[:, :, index]
                result = result[:, :, tf.newaxis]
                tensors.append(result)
            return tf.concat(tensors, axis=-1)
        
        result = inputs[:, :, self.label_index]  # return the input for a given target variable
        return result[:, :, tf.newaxis]


class MultiStepLastBaseline(Model):
    
    '''
    Create a baseline class, for Multi-Step Last Baseline. It means, instead of one steap ahead,
    we're froecasting of the next n-steps, given an n-input of data.

    Attributes
    ----------
    label_index : list
        Specify a list of targets we want to predict.

    Methods
    -------
    call(inputs):
        Return the data as TensorFlow instance.
    '''
    
    def __init__(self, label_index=None, steps=1):
        super().__init__()
        self.label_index = label_index
        self.steps = steps
    
    def call(self, inputs):
        if self.label_index is None:
            return tf.tile(inputs[:, -1:, :], [1, self.steps, 1])
        return tf.tile(inputs[:, -1:, self.label_index:], [1, self.steps, 1])
    

class RepeatBaseline(Model):
    
    '''
    Create a baseline class, for Multi-Step Repeat Baseline. In other words, let's say 
    if we set the steps to be 24 hours, then the prediction for the next 24 hours will be the 
    last known 24 hours of the data. 

    Attributes
    ----------
    label_index : list
        Specify a list of targets we want to predict.

    Methods
    -------
    call(inputs):
        Return the data as TensorFlow instance.
    '''
    
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
    
    def call(self, inputs):
        return inputs[:, :, self.label_index:]
    

def compile_and_fit(model, window, patience=3, max_epochs=50, save_model=False):
    '''
    Function to configure a deep learning, fit it on the data, and train the data.

    Parameters
    ----------
    model : 
        Pass the model from tensorflow.
    window : 
        Pass the DataWindow() instance.
    patience : 
        Number of epochs after which the model should stop training
        if the validation loss doesn't improve.
        Set default at 3.
    max_epochs : 
        Maximum number of epochs to train the model.

    Returns
    -------
    history : 
        The trained model.
    '''
    
    # early stopping to stop the model from training if there's no imporvement in the loss function
    
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=patience,
                                   mode='min')
    
    # model compiler, to specify the loss function, optimizer, and the metrics
    
    model.compile(loss=MeanSquaredError(),
                  optimizer=Adam(),
                  metrics=[MeanAbsoluteError()])
    
    # fitting the model, callbacks to stop it early if the validation loss is not decreased after 3 (default set) consecutive periods
    
    history = model.fit(window.train,
                        epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    
    if save_model == True:
        return model, history
    
    else:
        history


class AutoRegressive(Model):  # again, inherit the Model method from tf.keras
    '''
    Create, train, and make the prediction of a time-series using 
    Auto-Regressive Long Short-Term Memorty method (ARLSTM).

    Attributes
    ----------
    units : int
        Number of units for the neural networks model layer.
    out_steps : int
        How many n-steps predictions ahead you want to get.
    train_df : pd.DataFrame()
        The training dataframe.

    Methods
    -------
    warmup(inputs):
        Split the window between inputs and labels.
    call(inputs, training=None):
        Plot the input data, prediction, and actual values.
    '''
    
    def __init__(self, units, out_steps, train_df):
        super().__init__()
        self.units = units
        self.out_steps = out_steps
        self.train_df = train_df
        self.lstm_cell = LSTMCell(units)                        # lower level of LSTM layer, to be able to obtain granular details such as state and outputs
        self.lstm_rnn = RNN(self.lstm_cell, return_state=True)  # RNN layer wraps the LSTM layer, to make the data raining easier
        self.dense = Dense(train_df.shape[1])                   # the prediction comes from this dense layer
        
    
    def warmup(self, inputs):
        '''
        Replicate the single-step LSTM model, to return
        the first step prediction and its state.

        Attributes
        ----------
        inputs:
            Pass the input.
        
        
        Return
        ------
        The first step of the prediction and the state. 
        '''
        
        # x, *state -> tuple unpacking. takes a tuple or a sequence of values and assigns them to variables in a single statement.
        # so for example if output is (1, 2, 3, 4, 5) -> x = 1 and state = [2, 3, 4, 5]
        x, *state = self.lstm_rnn(inputs)  
        prediction = self.dense(x)
        
        return prediction, state
    
    
    def call(self, inputs, training=None):
        '''
        Loop the prediction and generate the sequence of predictions
        based on the out_steps parameter you define previously.

        Attributes
        ----------
        inputs:
            Pass the input.
        training:
            Parameter for the LSTM cell, inherited from tf.keras
        
        
        Return
        ------
        All the predictions. 
        '''
        
        predictions = []
        prediction, state = self.warmup(inputs)
        
        predictions.append(prediction)
        
        for n in range(1, self.out_steps):
            x = prediction
            x, state = self.lstm_cell(x, states=state, training=training)
            
            prediction = self.dense(x)
            predictions.append(prediction)
        
        # related to the data structure of tensorflow
        # see the documentations of the tensoreflow data
        
        predictions = tf.stack(predictions)
        predictions = tf.transpose(predictions, [1, 0, 2])  
        
        return predictions


def plot_loss_compare(dic, labels, **kwargs):
    '''
    Make a plot to turn our previously defined dictionary into a plot. 
    To make a comparison of the loss functions from every model we built.

    Parameters
    ----------
    dic : 
        Pass the dictionary that has been defined before that contains the loss value.
    labels : list
        Pass the label for the barplot, respectively by the order.
    **kwargs:
        Pass the parameteres from pd.DataFrame.plot() method.

    Returns
    -------
    matplotlib.pyplot figure from plt.show()
    '''
    
    # create the bar chart
    fig, ax = plt.subplots()
    temp = pd.DataFrame(dic).T.rename(columns={0:labels[0], 1:labels[1]})
    temp.plot(kind='bar', ax=ax, **kwargs)

    # add annotations to the chart
    for i, row in enumerate(temp.T.values):
        for j, val in enumerate(row):
            ax.text(j + i * 0.2, val, str(round(val, 4)), horizontalalignment='center', verticalalignment='bottom')

    # show the chart and remove the spines
    sns.despine()
    plt.show()


def prepare_data_for_prediction(df, window_size, target_column):
    '''
    Prepare dataset for model.predict(). Need to preprocessed first
    as the data is trained on a DataWindow() class object. Thus, this
    function looks very similar to the DataWindow class.

    Attributes
    ----------
    df:
        Dataframe that contains the data we want to predict.
    window_size:
        Window size of the DataWindow() class previously. Not really
        affecting that much if we already know what we're going to do.
        
        
    Return
    ------
    data : array
        Array data which is ready to be passed to the model. Contain either target or predictor variables.
    target :
        Only contains array data for the target variables.
    '''
    
    # get the target column as a separate Series
    target = df[target_column].values

    # create a list to hold the preprocessed windows
    data = []

    # iterate over the DataFrame, creating a window for each row
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size].values
        data.append(window)

    # convert the list of windows to a numpy array
    data = np.array(data)

    # reshape the data to fit the input shape of the model
    data = data.reshape(-1, window_size, df.shape[1])

    return data, target