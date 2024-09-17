#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Using the past 60 days stock price
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import pandas_datareader as web 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')


# In[2]:


# Load the data from the CSV file
df = pd.read_csv('AAPL-2.csv')

# Display the data
print(df)


# In[3]:


#get rows and columns
df.shape


# In[4]:


#visualize yhe closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Prize USD($)', fontsize = 18)
plt.show()


# In[5]:


#create  new dataframe with only the 'Close column'
data = df.filter(['Close'])
#convert the dataset to a numpy array
dataset = data.values
#get the number of rows to train the model on
training_data_len = math.ceil(len(dataset)* .8)
training_data_len


# In[6]:


#scale the data
scaler = MinMaxScaler(feature_range = (0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[7]:


#Create the training data set
#Create the scaked trining data set
train_data = scaled_data[0:training_data_len, :]

#Split the data into x_train and y_train data set
x_train = [] #independent training variables
y_train = [] #dependent training variables

for i in range(60, len(train_data)):
    x_train.append(train_data[i - 60:i, 0]) #will contain 60 values from index 0 to 59
    y_train.append(train_data[i, 0]) #will contain 61st value on index 60
    
    if i<=61:
        print(x_train)
        print(y_train)
        print()


# In[8]:


#Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)


# In[9]:


#Reshape the data
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[10]:


#Built the LSTM model
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape =( x_train.shape[1], 1)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(50))
model.add(Dense(1))


# In[11]:


#Compile the model
model.compile(optimizer='adam', loss ='mean_squared_error') 
#the optimizer is used to improve upon the loss funtcion
# the loss function is used to measure how well the model did on training


# In[12]:


#Train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)
# epoch the number of entire iterations 


# In[13]:


#Create the testing data set
#Create a new array containing scaled values from index 140 to 200
test_data = scaled_data[training_data_len - 60: , :]

#Create the data sets x_test and y_test
x_test =[]
y_test = dataset[training_data_len:, :] 
#y_test are all of the values that we want our model to predict

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])


# In[14]:


#Convert the data into numpy array
x_test = np.array(x_test)


# In[15]:


#Rehsape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


# In[16]:


#Get the models predicted the price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions) #unsclaing the values


# In[17]:


#Getting the root mean squared error (RMSE) 
#(the standard deviation of the residuals)
#the lower value of RMSE indicate a better fit (0 means that the predictions are perfect)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse 


# In[18]:


#Plot the data
train = data[:training_data_len].copy()
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions[:len(valid)]

#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Data', fontsize = 18)
plt.ylabel('Close Prise USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show


# In[19]:


#Show the valid and predicted prices
valid


# In[20]:


#Get the quote
selected_data = pd.read_csv('/kaggle/input/appl-data-2/AAPL-2.csv')

# Set the start and end dates
start_date = '2012-01-01'
end_date = '2024-03-01'

# Filter the data for the specified period
apple_quote = selected_data[(selected_data['Date'] >= start_date) & (selected_data['Date'] <= end_date)]

#Create a new dataframe
new_df = apple_quote.filter(['Close'])

#Get the last 60 days closing price and convert the dataframe to an array
last_60_days = new_df[-60:].values

#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

#Create an empty list
X_test = []
#Append the past 60 days
X_test.append(last_60_days_scaled)

#Convert the X_test data set to a numpy array
X_test = np.array(X_test)

#Reshape the data
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Get the predicted scaled price
pred_price = model.predict(X_test)

#Undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)


# In[21]:


selected_data = pd.read_csv('/kaggle/input/appl-data-2/AAPL-2.csv')

# Set the start and end dates
start_date = '2024-03-01'
end_date = '2024-03-01'

# Filter the data for the specified period
filtered_data = selected_data[(selected_data['Date'] >= start_date) & (selected_data['Date'] <= end_date)]

# Display the 'Close' prices for this period
print(filtered_data['Close'])

