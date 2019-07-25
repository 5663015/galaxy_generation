import numpy as np
import tensorflow as tf
from scipy.stats import norm
from keras.models import Model, load_model
from keras.layers import Layer, Dense, Reshape, Lambda,Conv2D
from keras import Input
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.objectives import binary_crossentropy
from keras.utils import multi_gpu_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from PIL import Image
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

config_ = tf.ConfigProto()  
config_.gpu_options.per_process_gpu_memory_fraction = 0.9  # 程序最多只能占用指定gpu50%的显存  
config_.gpu_options.allow_growth = True      #程序按需申请内存  
sess = tf.Session(config = config_)
with tf.Session(config=config_) as sess:
	pass

image_size = 256
latent_dim = 4
batch_size = 64

# 自定义层
class CustomVariationalLayer(Layer):
	def vae_loss(self, x, z_decoded, z_mean, z_log_var):
		x_ = K.flatten(x)
		z_decoded_ = K.flatten(z_decoded)
		xent_loss = binary_crossentropy(x_, z_decoded_)
		#xent_loss = K.mean(K.square(z_decoded_ - x_), axis=-1)
		kl_loss = -0.0001 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
		return K.mean(xent_loss + kl_loss)

	def call(self, inputs):
		x = inputs[0]
		z_decoded = inputs[1]
		z_mean = inputs[2]
		z_log_var = inputs[3]
		loss = self.vae_loss(x, z_decoded, z_mean, z_log_var)
		self.add_loss(loss, inputs=inputs)
		return x

def create_model():
	# 编码器网络
	input = Input(shape=(image_size, image_size, 3))
	x = Conv2D(64, (3, 3), padding='same', strides=2)(input)
	x = Activation('relu')(x)

	x = Conv2D(128, (3, 3), padding='same', strides=2)(x)
	x = Activation('relu')(x)

	x = Conv2D(256, (3, 3), padding='same', strides=2)(x)
	x = Activation('relu')(x)

	x = Conv2D(512, (3, 3), padding='same', strides=2)(x)
	x = Activation('relu')(x)

	shape_before_flattening = K.int_shape(x)
	x = Flatten()(x)

	# 均值方差
	z_mean = Dense(latent_dim)(x)
	z_log_var = Dense(latent_dim)(x)

	# 潜在空间采样的函数
	def sampling(args):
		z_mean, z_log_var = args
		epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1.)
		return z_mean + K.exp(z_log_var) * epsilon
	z = Lambda(sampling)([z_mean, z_log_var])

	# 解码器网络
	decoder_input = Input(K.int_shape(z)[1:])
	x = Dense(np.prod(shape_before_flattening[1:]), activation='relu')(decoder_input)
	x = Reshape(shape_before_flattening[1:])(x)

	x = UpSampling2D()(x)
	x = Conv2D(512, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	x = UpSampling2D()(x)
	x = Conv2D(256, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	x = UpSampling2D()(x)
	x = Conv2D(128, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	x = UpSampling2D()(x)
	x = Conv2D(64, (3, 3), padding='same')(x)
	x = Activation('relu')(x)

	x = Conv2D(3, (3, 3), padding='same', activation='tanh')(x)

	encoder = Model(input, z)
	decoder = Model(decoder_input, x)
	z_decoded = decoder(z)

	y = CustomVariationalLayer()([input, z_decoded, z_mean, z_log_var])
	model = Model(input, y)
	model.summary()
	return model, decoder, encoder

def plot_images(decoder):
	n = 8
	figure = np.zeros((image_size * n, image_size * n, 3))
	grid_x = norm.ppf(np.linspace(0.05, 0.85, n))
	grid_y = norm.ppf(np.linspace(0.05, 0.85, n))
	for i, yi in enumerate(grid_x):
		for j, xi in enumerate(grid_y):
			z_sample = np.array([[xi, yi, 0., 0.]])
			z_sample = np.tile(z_sample, batch_size).reshape(batch_size, latent_dim)
			x_decoded = decoder.predict(z_sample, batch_size=batch_size)
			digit = x_decoded[0].reshape(image_size, image_size, 3)
			figure[i * image_size: (i + 1) * image_size, 
					j * image_size: (j + 1) * image_size, :] = digit
	return figure


def train():
	# read data
	train_datagen = ImageDataGenerator(
		rescale=1./255,
		rotation_range=10, 
		horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
		'./data/train2/',
		target_size=(image_size, image_size),
		batch_size=batch_size,
		class_mode=None)
	# create model
	lr = 1e-3
	model, decoder, encoder = create_model()
	adam = Adam(lr)

	model = multi_gpu_model(model, gpus=2)
	model.compile(optimizer=adam, loss=None)
	
	index = 0
	epochs = 10000
	for image_batch in train_generator:
		index += 1
		time1 = time.clock()
		loss = model.train_on_batch(image_batch, None)

		if index % 100 == 0:
			print('[%d]/[%d]================' % (index, epochs))
			# plot generated images 
			image = plot_images(decoder)
			image = image*127.5+127.5
			Image.fromarray(image.astype(np.uint8)).save(
				'./figures5/' + str(index)+".png")
			print('loss:', loss)
			time2 = time.clock()
			print('running time:', str((time2 - time1)/60))
			# decrease learning rate
			p = float(index) / epochs
			lr = 0.001 / (1 + 20 * p) ** 0.75
			K.set_value(model.optimizer.lr, lr)

		if index == epochs:
			break

	decoder.save('decoder.h5')


if __name__ == '__main__':
	train()

