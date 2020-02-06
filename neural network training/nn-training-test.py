#!/usr/bin/env python
# coding: utf-8

# # D&D Dice Fraud Detection Software: Neural Network Training

# ## 0. Import

# NOTE: Make sure to switch Python / Anaconda environments to the TensorFlow one.

# In[1]:


import numpy as np
import tensorflow as tf
print('This is the Dice Fraud Detection Neural Network training script.')


# ## 1. Load Dataset and Train/Validate/Test Split

# In[2]:


with np.load("allDataShuffled.npz") as file:
    inputs = file['inputs']
    targets=file['targets']
Ntot=np.shape(targets)[0]
print('File loaded successfully, now training on a total of {0} datapoints'.format(Ntot))

#loads the file from the .npz, the savez method allows us to save multiple arrays with a key we define
#which is handy when saving objects coming from pandas DataFrames


# In[3]:


TRAIN_DATA_FRACTION=0.8
VALIDATE_DATA_FRACTION=0.1
TEST_DATA_FRACTION=0.1
#defines the fractions used for train/test/splitting. Note that these are technically model metaparameters.


# In[4]:


train_inputs = inputs[:int(np.ceil(TRAIN_DATA_FRACTION *Ntot))]
validate_inputs = inputs[int(np.ceil(TRAIN_DATA_FRACTION *Ntot)):int(np.ceil(VALIDATE_DATA_FRACTION*Ntot))]
test_inputs=inputs[int(np.ceil(VALIDATE_DATA_FRACTION*Ntot)):]

train_targets = targets[:int(np.ceil(TRAIN_DATA_FRACTION *Ntot))]
validate_targets = targets[int(np.ceil(TRAIN_DATA_FRACTION *Ntot)):int(np.ceil(VALIDATE_DATA_FRACTION*Ntot))]
test_targets=targets[int(np.ceil(VALIDATE_DATA_FRACTION*Ntot)):]
print('Batching process complete')
#perform train/validate/test splitting, we assume the data is already shuffled.


# ## 2. Define Neural Network and its Metaparameters

# Do not, I repeat, do **NOT** mess with the Metaparameters and run the training on the same data set. Create an entirely new batch of data points with the data generation script, to avoir all possibility of overfitting.

# In[5]:


INPUT_SIZE=6
#This is fixed by our requirements: one dice set is 6 numbers

HIDDEN_LAYER_SIZE=20
#height of each hidden layer.
TRAIN_RATE=0.001
#train rate can be small if using an early stopping mechanism
OPTIMIZER=tf.keras.optimizers.Adam(learning_rate=TRAIN_RATE)
#Adam has a momentum term that accelerates the computation but is bad for shallow minima. 
HIDDEN_LAYER_NUMBER=5
#Number of hidden layers before the final output layer
LAYER_SEQUENCE=[tf.keras.layers.Dense(HIDDEN_LAYER_SIZE,activation='relu') for i in range(HIDDEN_LAYER_NUMBER)]
#Defines the hidden layer sequence based on previous parameters. 
#Activation functions are all Rectified Linear Unit, which causes fewer problems than a basic sigmoid
#But requires normalisation, hence:
OUTPUT_SIZE=2
LAYER_SEQUENCE.append(tf.keras.layers.Dense(OUTPUT_SIZE, activation='softmax'))
#Final output layer using a softmax i.e. sigmoidal output to create a unit vector
LOSS='sparse_categorical_crossentropy'
#the correct loss function for a one-hot encoded output layer, i.e. a categorisation problem

#Note that technically we only have one output category by elimination, the following is more suited for a binary problem
#OUTPUT_SIZE=1
#LAYER_SEQUENCE.append(tf.keras.layers.Dense(OUTPUT_SIZE, activation='sigmoid'))
#LOSS='binary_crossentropy'

METRICS=['accuracy']
#get feedback for the model's accuracy as training goes on
BATCH_SIZE=100
#to match with the size of the dataset. 
MAX_EPOCHS=200
#to match with the train rate
CALLBACKS= tf.keras.callbacks.EarlyStopping(patience=2)
#sets up an early stopping mechanism that interrupts training if loss stops decreasing.
#patience parameter allows for some fluctuation up and down of the parameter before a full stop, useful for batch training



# In[6]:


nnmodel=tf.keras.Sequential(LAYER_SEQUENCE)

nnmodel.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
print('Model compiled, ready to train')


# ## 3. Train Network

# In[ ]:


nnmodel.fit(train_inputs,train_targets,batch_size=BATCH_SIZE,epochs=MAX_EPOCHS,
          callbacks=[CALLBACKS],
          validation_data=(validate_inputs,validate_targets))


# ## 4. Test Model

# ## 5. Export Model

# In[ ]:


5+5


# In[ ]:




