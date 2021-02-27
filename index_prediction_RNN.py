import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler

# Reading dataset
def read_data():
    input_data = np.array(pd.read_csv('EUR_USD_Weekly.csv', usecols = ['Price', 'Open', 'High', 'Low']))
    out_put_data = np.array(pd.read_csv('EUR_USD_Weekly.csv', usecols = ['Next High']))
    return input_data, out_put_data

# Standarizing dataset
def data_standarization(input):
    scaler = StandardScaler()
    scaler.fit(input)
    input = scaler.transform(input)
    return input, scaler

# Sequencing data to feed into the RNN
def sequencing_data(input, output):
    x = np.reshape(input, (input.shape[0], 1, input.shape[1]))
    y = np.reshape(output, (output.shape[0], 1, output.shape[1]))
    return x, y

# Splitting dataset linearly to preserve temporary correlation
def train_test_linear_split(x,y, train_ratio):
    train_boundary = int(train_ratio*len(x))
    train_x = x[0:train_boundary]
    test_x = x[train_boundary:]
    train_y = y[0:train_boundary]
    test_y = y[train_boundary:]
    return train_x, test_x, train_y, test_y

# Helper method to extract a portion of data
def data_subset(x, y, data_percentage):
    data_boundary = int(len(x)*data_percentage)
    x = x[0: data_boundary]
    y = y[0: data_boundary]
    return x, y 

# training and compiling model by providing train data and test data along with the timescale,
# the name of the index that is being approximate, the number of epochs our model and the learning rate
def train_model(train_x, test_x, train_y, test_y, timescale, index_name, number_epochs = 500, learning_rate = 0.005):
    loss_function = "huber_loss"
    model = Sequential()
    model.add(LSTM(128, input_shape = train_x.shape[1:], activation = 'relu', return_sequences = True))
    model.add(LSTM(64, activation = 'relu', return_sequences = True))
    model.add(LSTM(32, activation = 'relu', return_sequences = True))
    model.add(LSTM(8,  activation  = 'relu', return_sequences = True))
    model.add(Dense(4, activation = 'relu'))
    model.add(Dense(1, activation = 'relu'))
    opt = Adam(lr= learning_rate, decay=1e-5)
    model.compile(loss= loss_function, optimizer=opt)
    model.fit(train_x, train_y, validation_data = (test_x, test_y), epochs = number_epochs)
    model.save(index_name + '_' + timescale + '_RNN_epochs_' + str(number_epochs) + '_Learning_Rate_' + str(learning_rate) +'.h5') 
    return model

# Plotting 
def plot(x_axis, y_axis, title, x_label, y_label):
    for axis in y_axis:
        plt.plot(x_axis, axis['y_axis'], color = axis['color'], linestyle = '-', marker = '.' , label = axis['label']) #Lines
    plt.title(title) #Title of Plot
    plt.xlabel(x_label) #Label for x axis
    plt.ylabel(y_label) #Label for y axis
    plt.grid(True) #Deploying grid
    plt.legend() #Deploying legends
    plt.show()

# Plotting the index approximation vs actual index over the testing set
def plotting_approximation(test_y, predicted_y):
    test_y = np.reshape(test_y, (len(test_y),))
    predicted_y = np.reshape(predicted_y, (len(predicted_y),))

    percentage_error = 100*np.abs((test_y - predicted_y)/test_y)
    x = range(len(test_y))

    plot(x ,[{'y_axis': test_y, 'color': 'g', 'label': 'real'}, 
        {'y_axis': predicted_y, 'color': 'r', 'label': 'predicted'}],
        'Index', 'Days', 'High Price')

    plot(x, [{'y_axis' :percentage_error, 'color': 'b', 'label': 'percentage error'}], 'Percentage Error', 'Days', '%')

def approximating(input_features, data_scaler, model):
    for input in input_features:
        input = np.array(input)
        input = np.reshape(input,(1, -1))
        scaled_input = data_scaler.transform(input)
        approximation = model.predict([[scaled_input]])
        print(f"Close: {input[0][0]}  Open: {input[0][1]}  High: {input[0][2]}  Low: {input[0][3]}")
        print(f"Approximation: {approximation}")

non_standard_x, y = read_data()
x, data_scaler = data_standarization(non_standard_x)
x, y = sequencing_data(x,y)
train_x, test_x, train_y, test_y = train_test_linear_split(x,y, 0.70)

model = train_model(train_x, test_x, train_y, test_y, 'Weekly', 'EURUSD', number_epochs = 2000, learning_rate = 0.00001)
predicted_y = model.predict(test_x)
plotting_approximation(test_y, predicted_y)
