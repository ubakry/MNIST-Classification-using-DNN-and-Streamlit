### MNIST Classification using DNN and Streamlit
### © Copyright 2021 Usama Bakry (u.bakry@icloud.com)

## Import packages 
import streamlit as st
import tensorflow as tf
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import sys
import pickle
import gzip
import pandas as pd
import matplotlib.pyplot as plt

# Set page configurations
st.set_page_config(page_title='MNIST Classification using Deep Neural Network')

# Set the title 
st.title('MNIST Classification using Deep Neural Network')
'© Copyright 2021 Usama Bakry (u.bakry@icloud.com)'

# Set the number of neurons, the number of epochs, and the activation number
num_neurons = st.sidebar.slider("Number of neurons in the hidden layer:",1,64)
num_epochs = st.sidebar.slider("Number of epochs",1,10)
activation = st.sidebar.selectbox(
    "Activation function",
    ("relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential")
)

# Train the model
if st.button('Train the model'):

    "Please wait..."

    # Load MNIST data
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    if sys.version_info < (3,):
        data = pickle.load(f)
    else:
        data = pickle.load(f, encoding='bytes')
    f.close()

    # Split the data to training and test sets
    (X_train,y_train),(X_test,y_test) = data

    # Preprocess the image function
    def preprocess_image(images):
        images =images/255
        return images

    # Preprocess the training and test images
    X_train = preprocess_image(X_train)
    X_test = preprocess_image(X_test)

    # Build the neural network
    model = Sequential()
    model.add(InputLayer((28,28)))
    model.add(Flatten())
    model.add(Dense(num_neurons,activation))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # Save the model check point
    save_cp = ModelCheckpoint('model',save_best_only=True)
    output_cp = CSVLogger('output.csv',separator=',')

    # Run the model
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=num_epochs,callbacks=[save_cp,output_cp])

    "Done. Click on \"Evaluate the model\" button."

# Evaluate the model
if st.button('Evaluate the model'):

    # Read the output file
    output = pd.read_csv('output.csv')

    # View input parameters
    st.header("Input parameters:")
    st.markdown("Number of neurons in the hidden layer: **" +str(num_neurons)+"**")
    st.markdown("Number of epochs: **" + str(num_epochs)+"**")
    st.markdown("Activation function: **" + activation+"**")

    # View output measures
    st.header("Output measures:")
    st.dataframe(output)

    # Plot the figure accuracy vs apochs
    fig = plt.figure()
    plt.plot(output['epoch'],output['accuracy'])
    plt.plot(output['epoch'],output['val_accuracy'])
    plt.title('Model accuracy vs epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train','Val'])
    fig


