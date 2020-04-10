# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 22:58:01 2020

@author: simon
"""

import numpy as np
import pandas as pd
import os
import re
from keras.layers import LSTM,Dense, TimeDistributed, Reshape, Lambda, Input
from keras.models import Model
import keras
from keras.utils import np_utils



observed_days = 60
future_days = 30

test_days = 50

train_pct = 0.86
num_samp_per_stock = 20
min_pct_increase = 0.1 # this is equivilant to 10%



#Read data
df_merge = pd.read_csv("./stockProject/data/ML_merge.csv")
#Drop redundant columns
df_merge = df_merge.drop(['Last', 'Volume', 'Last.1', 'Market.Capitalization.1', 'Price.to.Earnings.Ratio..TTM..1',
                          'Basic.EPS..TTM..1', 'Last.2', 'Basic.EPS..TTM..2', 'EPS.Diluted..FY..1'], axis=1)

Names = df_merge['Ticker']


# Convert Sector column to categorical vector
sectorToCategorical =  np_utils.to_categorical(df_merge.Sector.factorize()[0])
sectorNames = df_merge['Sector'].unique()
for i in range(0, len(sectorNames)):
    sector = sectorNames[i]
    df_merge[sector] = sectorToCategorical[:,i]
df_merge = df_merge.drop(['Sector'], axis = 1)





# This function is used to generate training samples from time series data
# observed_days is the time frame in which the ML algorithm is allowed to see
# future_days is the amount of days to look into the future to compute for the pct_increase from today's price
# num_samp is the number of samples generated from each stock
# pct_increase is the minimum amount of increased in percent change to in order for the sample to be considered True, anything below is False
# threshold_days is the minimum number of days such that the stock price in the future_days is above threshold
# only stocks that maintains a stock price above pct_increase for a minimum of threshold_days is considered a True examples

def generateSamples(df, observed_days=60, future_days = 30, num_samp = 10, pct_increase = 0.1, threshold_days = 3):
    
    X = []
    Y = []
    todayVal = []
    lastDayVal = []
    pctGain = []
    for i in range(0, num_samp):
        index = np.random.randint(0,df.shape[0]-future_days - observed_days,1)[0]
        current_df = df.iloc[index:(index+observed_days),:]
        X.append(current_df.values)
        
        current_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted']
        
        altered_df = df.iloc[(index+observed_days):(index+observed_days+future_days),:]
        altered_df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adjusted']
        
        altered_df = altered_df.sort_values(by='Close', ascending = False)
        
        value_future = altered_df.iloc[threshold_days -1 ,3]
        value_today = current_df.iloc[-1,3]
        
        if value_future > value_today*(1+pct_increase):
            Y.append(True)
        else:
            Y.append(False)
        
        todayVal.append(value_today)
        lastDayVal.append(value_future)
        pctGain.append((value_future-value_today)/value_today)
            

    return X, Y, todayVal, lastDayVal, pctGain


def appendFinancialFeatures(X, row):
    X_financials = []
    for i in range(0, len(X)):
        X_financials.append(row)

    return X_financials

# convert [True, False] to [1, 0] and [0, 1] respectively
def toCatagorical(Y):
    retY = []
    for element in Y:
        if element:
            retY.append(np.array([1,0]))
        else:
            retY.append(np.array([0,1]))
    return np.array(retY)


def subsampleToBalance(X, Y, size = 5000):
    # Y must be a catagorical vector
    
    class1 = np.where(Y[:,0] == 1)[0]
    class2 = np.where(Y[:,0] == 0)[0]
    
    samples1 = np.random.choice(class1, size, True)
    samples2 = np.random.choice(class2, size, True)
    
    all_samples = np.concatenate([samples1,samples2])

    return all_samples
    

def trainTestSplit(X, train_size = 0.8):
    num_samples = int( len(X) * 0.8)
    
    samples = np.random.choice(range(0,len(X)), num_samples, False)
    
    return samples
    

def normalizeInputX(X):
    
    for i in range(0,len(X)):

        X[i] = X[i]-X[i].mean(axis=0)
        var = X[i].var(axis = 0)
        if np.any(var):
            continue
        X[i] = X[i]/var
        
    return X
        
        

"""
for file in glob.glob("./stockProject/data/ML_dataset/downloadStocks/./*.csv"):
    print(file)
"""
# have to redo names
Names = os.listdir("./stockproject/data/ML_dataset/downloadStocks/")



X_train = []
Y_train = []
todayVal_train = []
lastDayVal_train = []
X_financials_train = []

X_test = []
Y_test = []
todayVal_test = []
lastDayVal_test = []
X_financials_test = []

for name in Names:
    

    df = pd.read_csv("./stockproject/data/ML_dataset/downloadStocks/" + name)
    
    myName = name
    
    
    m = re.search('^(.*)\.csv$', name)
    if m:
        myName = m.group(1)
    else:
        print("error encountered")
        continue
    #print("here2")
    myName = m.group(1)

    
    if df.shape[0] <= observed_days + future_days + test_days:
        continue
    
    
    # simulate a random coin flip to either put the data for this stock in training or testing data
    coin_flip = np.random.uniform(0,1,1)[0]
    
    if coin_flip <= train_pct:
        
        # remove test days from df
        df = df.iloc[:-test_days,:]
        
        X, Y, todayVal, lastDayVal, pctGain = generateSamples(df, future_days = future_days, num_samp = num_samp_per_stock, pct_increase = min_pct_increase)
        currentRow = df_merge.loc[df_merge['Ticker'] == myName].values    
        if len(currentRow) == 0:
            continue    
        X_financials = appendFinancialFeatures(X, currentRow)

        
        X_train = X_train +  X
        Y_train = Y_train + Y
        todayVal_train = todayVal_train + todayVal
        lastDayVal_train = lastDayVal_train + lastDayVal 
        X_financials_train = X_financials_train + X_financials
    else:
        
        # remove none test_days for evaluating pct_increase
        df = df.iloc[-(test_days+observed_days):,:]
        
        X, Y, todayVal, lastDayVal, pctGain = generateSamples(df, future_days = future_days, num_samp = num_samp_per_stock, pct_increase = min_pct_increase)
        currentRow = df_merge.loc[df_merge['Ticker'] == myName].values    
        if len(currentRow) == 0:
            continue    
        X_financials = appendFinancialFeatures(X, currentRow)
        
        
        X_test = X_test +  X
        Y_test = Y_test + Y
        todayVal_test = todayVal_test + todayVal
        lastDayVal_test = lastDayVal_test + lastDayVal 
        X_financials_test = X_financials_test + X_financials        
    

# Process training data 
X_train = np.array(X_train)
Y_train = np.array(Y_train)
todayVal_train = np.array(todayVal_train)
lastDayVal_train = np.array(lastDayVal_train)
X_financials_train = np.array(X_financials_train)

X_financials_train = X_financials_train[:,0,1:]
X_financials_train = X_financials_train.astype(float)

Y_train = toCatagorical(Y_train)
Y_train.sum(axis = 0)

#X_all = normalizeInputX(X_all)

# Process testing data
X_test = np.array(X_test)
Y_test = np.array(Y_test)
todayVal_test = np.array(todayVal_test)
lastDayVal_test = np.array(lastDayVal_test)
X_financials_test = np.array(X_financials_test)

X_financials_test = X_financials_test[:,0,1:]
X_financials_test = X_financials_test.astype(float)

Y_test = toCatagorical(Y_test)
Y_test.sum(axis = 0)


# Since the training data is highly imbalanced, where the number of positive examples is 
# far less than the number of negative examples, I will rebalance the data by random sampling 
# 50000 samples from each of the two classes such that I have 5000 positive examples and 5000 negative
# examples in the training data. Note, the sampling is sampling with replacement so repeated examples 
# are expected
balance_index = subsampleToBalance(X_train, Y_train, size = 5000)

X_train_B = X_train[balance_index,:,:]
Y_train_B = Y_train[balance_index]
todayVal_train_B = todayVal_train[balance_index]
lastDayVal_train_B = lastDayVal_train[balance_index]
X_financials_train_B = X_financials_train[balance_index,:]


"""
train_index = trainTestSplit(balance_index, train_size = 0.8)

X_all_train = X_all_B[train_index,:,:]
Y_train = Y_B[train_index]
todayVal_train = todayVal_B[train_index]
lastDayVal_train = lastDayVal_B[train_index]
X_financials_train = X_financials_B[train_index]
"""


def rnn(x_shape,  financial_shape ,  n_classes = 2):
    X = Input(x_shape)
    X_financials = Input(financial_shape)
    todayVal = Input((1,))

    net = LSTM(8, activation = 'relu', return_sequences = False)(X)
    #net = LSTM(8, activation = 'relu')(net)

    net = keras.layers.concatenate([net, X_financials, todayVal], axis = 1)
    net = Dense(128, activation = 'relu')(net)
    #net = Dense(32, activation = 'relu')(net)
	#net = Dropout(0.8)(net)
    

    out = Dense(2, activation = 'softmax')(net)

    print(out.shape)
    model = Model(inputs=[X, X_financials, todayVal],outputs=out)
    return model




model = rnn((X_train_B.shape[1], X_train_B.shape[2]), (X_financials_train_B.shape[1],))
print(model.summary())

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])





# Further refine training data by removing any NAs 
"""
if np.isnan(X_train_B).any():
    np.nan_to_num(X_train_B, 0)
    print("nans in X_all")
    
if np.isnan(X_financials_train_B.astype(float)).any():
    print("nans in X_financials_all")
"""    
if np.isnan(todayVal_train_B).any():
    np.nan_to_num(todayVal_train_B, 0)

if np.isnan(X_train_B).any():
    np.nan_to_num(X_train_B, 0)
    
if np.isnan(X_financials_train_B).any():
    np.nan_to_num(X_financials_train_B, 0)



model.fit([X_train_B, X_financials_train_B, todayVal_train_B[...,np.newaxis]], Y_train_B,
                    batch_size=64,
                    epochs=100)


model.save('stockClassifier2.h5')




# Process testing data
if np.isnan(todayVal_test).any():
    np.nan_to_num(todayVal_test, 0)

if np.isnan(X_test).any():
    np.nan_to_num(X_test, 0)
    
if np.isnan(X_financials_test).any():
    np.nan_to_num(X_financials_test, 0)


pred = model.predict([X_test, X_financials_test,  todayVal_test[...,np.newaxis]])





#Evaluate model performance on test data
from sklearn.metrics import confusion_matrix
Y_pred_classes = np.argmax(pred, axis = 1) 
Y_test_classes = np.argmax(Y_test, axis = 1) 
fn, tp, tn, fp = confusion_matrix(Y_test_classes, Y_pred_classes).ravel()

accuracy = (tp+tn)/(tp+tn+fp+fn)
print("test accuracy = " + str(accuracy))

sucess_rate = tp/(fp + tp)
print("percision = " + str(sucess_rate))


print(confusion_matrix(Y_test_classes, Y_pred_classes))

samples_predicted_true = np.where(Y_pred_classes == 0)[0]
tp_samples = np.where(  np.logical_and( Y_pred_classes == 0 , Y_test_classes == 0 ) )[0]
fp_samples = np.where(  np.logical_and( Y_pred_classes == 0 , Y_test_classes != 0 ) )[0]

earnings_from_fp_samples = (lastDayVal_test[fp_samples] - todayVal_test[fp_samples])/todayVal_test[fp_samples]


# avoid problems when dividing by zero
np.nan_to_num(earnings_from_fp_samples, 0)
#sanity check
np.where(earnings_from_fp_samples < -1) # this should find nothing
problems = np.where(earnings_from_fp_samples > min_pct_increase) # check for any unexpected outcomes
earnings_from_fp_samples[problems] # print them out.
# I observed some infinities

# set them to zero
earnings_from_fp_samples[problems] = 0 



earnings_from_fp_samples = np.nan_to_num(earnings_from_fp_samples, -1) # just in case if there are any Nans
total_change_from_fp_samples = earnings_from_fp_samples.sum()
average_pct_earnings = ( len(tp_samples) * min_pct_increase + total_change_from_fp_samples ) / len(samples_predicted_true)

print("Expected monthly percent earnings " + str(average_pct_earnings))


small_vals = np.where(todayVal_test < 0.1)[0] # ignore stocks that are less than 10 cents
today = np.delete(todayVal_test, small_vals)
last_day = np.delete(lastDayVal_test, small_vals)

Buy_and_Hold = (today - last_day)/today
Buy_and_Hold = np.nan_to_num(Buy_and_Hold, 0)
#Buy_and_Hold[np.isneginf(Buy_and_Hold)] = 0
print("Expected return from buy and hold = ", str(Buy_and_Hold.mean()))


# add time series predictions, LSTM ?
# use neural ODE for this ? 


#X_financials_test[samples_predicted_true[0:-1],-4].sum()
