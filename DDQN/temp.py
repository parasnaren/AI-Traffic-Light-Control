import keras
import keras.backend as K


def func(i):
	return K.expand_dims(i[:,0],-1) + i[:,1:] - K.mean(i[:,1:], keepdims=True)

func([1,2,3,4])
