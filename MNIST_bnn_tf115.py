#%% import tf, tfp, np, keras
import tensorflow as tf
#import keras.backend as K
import tensorflow_probability as tfp
import numpy as np
import random
import pandas as pd
#import newton_cg as es
import efficient_second as es




import matplotlib.pyplot as plt
import scipy.stats as stats

#%%

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))



#%%load and prepare data



n_classes = 10

(x_train, y_train), (x_test, y_test_dec) = tf.keras.datasets.mnist.load_data()

#rescale to [0.0,1.0]
x_train = x_train[..., np.newaxis]/255.0
x_test = x_test[..., np.newaxis]/255.0



y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test_dec, n_classes)


indices = list(range(x_train.shape[0]))
num_training_instances = int(0.8 * x_train.shape[0])
random.shuffle(indices)
train_indices = indices[:num_training_instances]
val_indices = indices[num_training_instances:]

x_val = x_train[val_indices]
y_val = y_train[val_indices]

x_train = x_train[train_indices]
y_train = y_train[train_indices]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))



#%%


np.random.seed(10)

# draw random samples for progress control
rand_ind = np.random.randint(0,9999,5)
rand_weights_mean = np.random.randint(0,803839,5)
rand_weights_var = rand_weights_mean + 803840

rand_weights_mean_output = np.random.randint(0,9,3)
rand_weights_var_output = rand_weights_mean_output + 10



#%% build/ model
#trainingset_batches = train_dataset.batch(128)  

train_size = x_train.shape[0]
test_size = x_test.shape[0]

input_shape=(28,28,1)
batch_size = 128
num_batches = train_size / batch_size
kl_loss_weight = 1.0 / num_batches

n_epochs = 20
noise=1

def prior_untrainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype, trainable = False),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
          tfp.distributions.Normal(loc=t, scale=1),
          reinterpreted_batch_ndims=1)),
    ])

def  posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.)) 
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1
            )),
    ])


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    tf.keras.layers.Flatten(),
    tfp.layers.DenseVariational(units=1024,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_untrainable,
                                kl_weight=kl_loss_weight,
                                activation='relu'),
    tfp.layers.DenseVariational(units=10,
                                make_posterior_fn=posterior_mean_field,
                                make_prior_fn=prior_untrainable,
                                kl_weight=kl_loss_weight,
                                activation='softmax')
])



    

learning_rate = 0.03

opt = es.EHNewtonOptimizer(learning_rate,tau=1e5)
#         #tau=FLAGS.eso_tau,
#         #=FLAGS.eso_cg_tol,
#         #max_iter=FLAGS.eso_max_iter)
#         )

#opt = tf.keras.optimizers.Adam(learning_rate)
#opt = tf.keras.optimizers.SGD(learning_rate)

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer = opt, metrics=[tf.keras.metrics.CategoricalAccuracy()])


#%%
y_rand_matrix = model.predict(x_test[rand_ind])
fig1, axs1 = plt.subplots(2, 1)
fig1.tight_layout()

bins=np.linspace(-1,1,100)
axs1[0].hist(model.get_weights()[0],bins, histtype='step')
axs1[0].set_title('Histogram of DenseVariational Hidden Layer Weights, 1024 units', y=1.2)
axs1[1].hist(model.get_weights()[2],bins, histtype='step')
axs1[1].set_title('Histogram of DenseVariational Output Layer Weights, 10 units')
fig1.subplots_adjust(hspace=0.6)
fig1.subplots_adjust(top=0.9)

var_before_training = np.var(model.get_weights()[0][:803839])
var_before_training_output = np.var(model.get_weights()[2][:10249])
print("Variance in weights (means), hidden layer, 1024 units, before training: " + str(var_before_training))

print("Variance in weights (means), output layer, 10 units, before training: " + str(var_before_training_output))




fig2, axs2 = plt.subplots(5, 1)

fig2.suptitle('Output of Initialized but Untrained Model for Random Samples')
for i in range(5):
    bins = [0,1,2,3,4,5,6,7,8,9]
    axs2[i].bar(bins,y_rand_matrix[i,:])
    axs2[i].set_ylim([0, 1.2])
    axs2[i].set_title("Sample no. " + str(rand_ind[i]) )
fig2.subplots_adjust(hspace=1.7)    
    
#%%    
trainingset_batches = train_dataset.batch(128) 

#%%

def train_step(x, y):
    with tf.GradientTape() as tape:
        y_log = model(x)
        loss = tf.keras.losses.categorical_crossentropy(y,y_log)
        loss += sum(model.losses) # kl divergence loss
    gradients = tape.gradient(loss, model.trainable_variables)
    
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss 

def scheduler(epoch, lr):
    if epoch < 5:
        return lr
    else:
        return lr * np.exp(-0.1)
    
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)


#%%
train_val_acc= np.zeros((n_epochs+1,2,2))
val_acc=model.evaluate(x_val, y_val, batch_size=128)
train_acc=model.evaluate(x_train, y_train, batch_size=128)
train_val_acc[0,0,1] = val_acc[0]
train_val_acc[0,1,1] = val_acc[1]
train_val_acc[0,0,0] = train_acc[0]
train_val_acc[0,1,0] = train_acc[1]



#%% loss/accuracy plot


# hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, callbacks=[callback], verbose=1, validation_data = (x_val,y_val));  
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs,  verbose=1, validation_data = (x_val,y_val));  
train_val_acc[1:,0,0]= hist.history['loss']    
train_val_acc[1:,0,1]= hist.history['val_loss']  
train_val_acc[1:,1,0]= hist.history['categorical_accuracy']    
train_val_acc[1:,1,1]= hist.history['val_categorical_accuracy']    



#%% plot acc graph
x_acc = np.linspace(0,n_epochs+1,n_epochs+1)
fig7, (axs1, axs2) = plt.subplots(2, 1)
#fig7.suptitle('Training and Validation Accuracy vs # of epochs')

axs1.plot(x_acc, train_val_acc[:,0,0], '-g', x_acc, train_val_acc[:,0,1], '-y')
axs1.legend(('train','val'))
axs1.set_title('Training and Validation Loss')
axs2.plot(x_acc, train_val_acc[:,0,0], '-g', x_acc, train_val_acc[:,0,1], '-y')

axs1.set_ylim(100, 350)  # outliers only
axs2.set_ylim(0, 80)  # most of the data

# hide the spines between ax and ax2
axs1.spines['bottom'].set_visible(False)
axs2.spines['top'].set_visible(False)
axs1.xaxis.tick_top()
axs1.tick_params(labeltop=False)  # don't put tick labels at the top
axs2.xaxis.tick_bottom()
axs2.set_xlabel('number of epochs')

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=axs1.transAxes, color='k', clip_on=False)
axs1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
axs1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=axs2.transAxes)  # switch to the bottom axes
axs2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
axs2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
fig7.subplots_adjust(hspace=0.3) 



fig8, ax = plt.subplots()
ax.plot(x_acc, train_val_acc[:,1,0], '-b', x_acc, train_val_acc[:,1,1], '-m')
ax.set_title('Training and Validation Categorical Accuracy')
#plt.gca().legend(('train','val'))
ax.legend(('train','val'),loc='lower right')
ax.set_xlabel('number of epochs')

        

  

#%% prediction

import tqdm

n_predictions = 50
y_pred_matrix = np.zeros((test_size,n_predictions,n_classes))
for i in tqdm.tqdm(range(n_predictions)):
    y_pred_matrix[:,i,:] = model.predict(x_test) 
    
y_pred_stat = np.zeros((test_size, 2,n_classes))
y_pred_dec = np.zeros(test_size)
list_unconfident =[]

for i in range(test_size):
    y_pred_stat[i,0,:]=np.mean(y_pred_matrix[i],axis=0)
    y_pred_stat[i,1,:]=np.std(y_pred_matrix[i], axis=0)
    y_pred_dec[i] = np.argmax(y_pred_stat[i][0])
    if (np.max(y_pred_stat[i][0]) <= 0.5):
        list_unconfident.append(i)

mean_class_std = np.mean(y_pred_stat[:,1,:],axis=0)
mean_class_std_unconfident = np.mean(y_pred_stat[list_unconfident,1,:], axis=0)
#%% accuracy

correct_pred = np.sum(y_pred_dec == y_test_dec)


acc=correct_pred/test_size

#%% print further information

