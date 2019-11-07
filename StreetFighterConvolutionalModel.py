import sys
sys.path.append("C:/Users/Oliver/Anaconda3/envs/gym/Lib/site-packages")
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import RMSprop

class street_fighter_convolutional:

    def __init__(self,input_shape, action_space):
        self.model = Sequential()
        self.model.add(Conv2D(input_shape=input_shape, filters=32, kernel_size=8, strides=4, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=3, strides=2, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(units=512, activation='relu'))
        self.model.add(Dense(units=action_space, activation='linear'))
        self.model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01), metrics=["accuracy"])
        self.model.summary()
