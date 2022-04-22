# import tensorflow as tf

import numpy as np
from tensorflow.keras import Input, Model  # , Sequential
# from tensorflow.keras.initializers import he_uniform
#import convolutional and Max pooling layers
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.regularizers import l2


def generate_model(num_classes,
					input_shape,
					num_filter = 32,
					act_func = "relu",
					kernel_size = (3, 3),
					layer_num = 3,
					drop_rate = 0.25):
	"""
	(num_classes,
				input_shape,
				num_filter = 32,
				act_func = "relu",
				kernel_size = (3, 3),
				layer_num = 3,
				drop_rate = 0.25,
				bath_norm = False)
	"""

	inputs = Input(input_shape)

	for i in range(1, layer_num+1):
		if i == 1:
			inptt = inputs
			x = Conv2D(i*num_filter, (5, 5),
							 strides=1,
							 input_shape = input_shape,
							 padding = 'same',
							 activation = act_func,
							 kernel_regularizer = l2(l=0.1),
							 kernel_initializer = 'he_normal')(inptt)
		else:
			inptt = x
			x = Conv2D((2*i-2)*num_filter, kernel_size,
							 strides=1,
							 input_shape = input_shape,
							 padding = 'same',
							 activation = act_func,
							 kernel_regularizer = l2(l=0.1),
							 kernel_initializer = 'he_normal')(inptt)
		# x = Conv2D(i*num_filter, kernel_size,
		# 					 input_shape = input_shape,
		# 					 padding = 'same',
		# 					 activation = act_func,
		# 					 kernel_initializer = he_uniform())(inptt)



		x = MaxPooling2D((2, 2), padding = 'same')(x)


		x = Dropout(drop_rate)(x, training=True)


	x = Flatten()(x)

	x = Dense((2*layer_num-2)*num_filter,
						 activation = act_func,
						 kernel_regularizer = l2(l=0.1),
						kernel_initializer = 'he_normal')(x)
	x = Dropout(drop_rate)(x, training=True)

	output = Dense(num_classes, kernel_regularizer = l2(l=0.1), activation = 'softmax')(x)

	return Model(inputs=inputs, outputs=output)



def get_epistemic_uncertainty(model, image, T=10):
	dim = image.shape
	p_hat = []
	epistemic = np.zeros(dim[0])
	for t in range(T):
		p_hat.append( model.predict(image) )

    # estimate uncertainties (eq. 4 )
    # eq.4 in https://openreview.net/pdf?id=Sk_P2Q9sG
	p_hat = np.array(p_hat)
    # mean prediction
	prediction = []
	p_bar = np.mean(p_hat, axis=0)
	for i in range(dim[0]):
		prediction.append(np.where( p_bar[i] > 0.5))
		for j in range(T):
			epistemic[i] += np.dot((p_hat[j,i] -p_bar[i]), (p_hat[j,i]-p_bar[i]))/T

	return prediction, epistemic
