import tensorflow.keras as keras
from tensorflow.keras import layers
import os
os.environ['TF_KERAS'] = '1'
import newton_cg as es
import argparse
#import horovod.keras as hvd



parser = argparse.ArgumentParser(description='Keras  regression case',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--optimizer', type=int, default=1, help='0 for ncg, 1 for adam, 2 for sgd')
args = parser.parse_args()

# hvd.init()

#initializer = keras.initializers.Zeros()

original_dim = 28 * 28
intermediate_dim = 64
latent_dim = 2

inputs = keras.Input(shape=(original_dim,))
h = layers.Dense(intermediate_dim, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)
z_log_sigma = layers.Dense(latent_dim)(h)

from keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mean, z_log_sigma])


# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

# Create decoder
latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = keras.Model(latent_inputs, outputs, name='decoder')

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = keras.Model(inputs, outputs, name='vae_mlp')



reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)


if args.optimizer<1:
    opt=es.EHNewtonOptimizer(0.01)
elif args.optimizer<2:
    opt=keras.optimizers.Adam(0.01)
else:
    opt=keras.optimizers.SGD(0.001)

# opt = hvd.DistributedOptimizer(opt)

vae.compile(optimizer=opt,loss='categorical_crossentropy',)




from keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
encoded_imgs = vae.predict(x_test)

vae.fit(x_train, x_train, epochs=100,batch_size=32, verbose=2, shuffle=True,validation_data=(x_test, x_test))

