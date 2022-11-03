
import argparse

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Enable tf v1 behavior as in v2 a lot have changed
# import dl_utils as utils
import newton_cg as es

print(tf.__version__)


parser = argparse.ArgumentParser(description='Keras  regression case',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--optimizer', type=int, default=0, help='0 for ncg, 1 for adam, 2 for sgd')

args = parser.parse_args()


df = pd.read_csv("HousingData.csv",header=0)
print(df)
#Get the value of df

df= df.dropna()

df = df.values

#Convert df to np array format
df = np.array(df)

#Characteristic data normalization
#Normalize (0-1) the column of feature data {0 to 11}
#for i in range(12):
#    df[:,i] = (df[:,i]-df[:,i].min())/(df[:,i].max()-df[:,i].min())
    
#x_data is the first 12 columns of feature data after normalization
x_data = df[:,:12]

#y_data is the last column of label data
y_data = df[:,12]

#x_data,y_data = shuffle(x_data,y_data)

#data = pd.read_csv("birth_rate.csv")
#data.head()
#print(data)

def sigmoid(x):

    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig

# Split data/labels
#data_X = np.array(data['Birth rate'])
data_X = x_data#  np.array(range(10))*0.1-0.5# np.array(data['Birth rate'])
data_Y = y_data # sigmoid(2*data_X +0.2) # np.array(data['Life expectancy'])
#data_Y =  np.array(data['Life expectancy'])

print(data_X)
print(data_Y)

#print(len(data))

#X = tf.placeholder(tf.float32, name='X')
#Y = tf.placeholder(tf.float32, name='Y')

#w = tf.get_variable('weight', initializer=tf.constant(0.1))
#b = tf.get_variable('bias', initializer=tf.constant(0.1))

#Y_hat = tf.add(tf.multiply(X,w), b)

#loss = (Y-Y_hat)*(Y-Y_hat) # tf.keras.losses.mean_squared_error(Y,Y_hat)
#loss = huber_loss(Y,Y_hat)

## Define gradient descent as the optimizer to minimise the loss
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
##optimizer = es.EHNewtonOptimizer(0.001).minimize(loss)


#model = keras.Model(inputs=[X], outputs=Y, name=f'model{i}')
#model = tf.keras.Sequential([
#    X,
#    layers.Dense(units=1)
#])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1))##,activation=tf.nn.sigmoid ))
model.build((None, 12))
print(model.get_weights())
if args.optimizer<1: 
    model.compile(optimizer=es.EHNewtonOptimizer(0.1), loss='mse')
elif args.optimizer<2:
    model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='mse')
else: 
    model.compile(optimizer=tf.train.GradientDescentOptimizer(1e-9), loss='mse')
sess = tf.Session()
sess.run(tf.global_variables_initializer())

model.summary()
model.evaluate(data_X, data_Y)
#model.compile(optimizer=tf.train.GradientDescentOptimizer(1), loss='mse')
for i in range(200):
    print(model.get_weights())
    model.fit(data_X, data_Y,  epochs=1)

print(model.trainable_variables) 

print(model.get_weights())

#sess = tf.Session()

history = []
start = time.time()

# Graph variable initialization
#sess.run(tf.global_variables_initializer())


# Open stream for tensorboard
#writer = tf.summary.FileWriter(logdir, sess.graph)

# Start training
#for i in range(50):
#    total_loss = 0.0
#    for x in range(len(data)):
#        _, l = sess.run([optimizer,loss], feed_dict={X: data_X[x], Y:data_Y[x]})
#        total_loss += l
#    if (i) % 10 == 0:
#        dw, db = sess.run([w,b])
#        y_hat = data_X * dw + db
#        history.append(y_hat)

#        print('Epoch {0}: {1}, w: {2}, b: {3}'.format(i, total_loss/len(data), dw, db))
        #print('dw: %f, db: %f\n' %(dw, db))

#writer.close()

#print('Train Time: %f seconds' %(time.time() - start))

