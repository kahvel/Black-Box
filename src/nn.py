import callback, callback2

import numpy as np
import itertools
import datetime
import random

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras import regularizers

# import entropy_estimators
# import pydot
# print(pydot.find_graphviz())
model = Sequential()

weight_decay = 0#0.0001
first_layer_decay = weight_decay
kernel_initialisation = "glorot_uniform"
add_label_input = False
use_two_xors = False
use_randomly_generated_inputs = False
sample_from_all_inputs = False
no_of_samples = 4096
directory = "./test3_4"
epochs = 5000
frequency = epochs/10
batch_size = 100

input_dim = 13 if add_label_input else 12
model.add(Dense(12, kernel_regularizer=regularizers.l2(first_layer_decay), kernel_initializer=kernel_initialisation, input_dim=input_dim))
model.add(Activation("tanh"))
model.add(Dense(10, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(7, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(5, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(4, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(3, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(2, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("tanh"))
model.add(Dense(1, kernel_regularizer=regularizers.l2(weight_decay), kernel_initializer=kernel_initialisation))
model.add(Activation("sigmoid"))
# model.add(Activation("tanh"))
# model.add(Dense(1))

model.compile(#loss='categorical_crossentropy',
              loss='binary_crossentropy',
              optimizer='sgd',)
              #metrics=['accuracy'])
print("Compiled model")
# plot_model(model, to_file='model.png')


def f(x):
    return sum(x) % 2
    # y = [0,0,1,1,0,1,None,None,None,None,None,None]
    # return int(all(map(lambda a: ((a[0] == a[1]) if a[1] is not None else True), zip(x, y))))


def f1(x):
    return sum(x[:6]) % 2


def filtering(x):
    return (sum(x[:6]) % 2) == (sum(x[6:]) % 2)


if use_randomly_generated_inputs:
    x_train = np.array([np.random.random_integers(0, 1, 12) for _ in range(no_of_samples)])
else:
    data_generator = itertools.product([0, 1], repeat=12)
    x_train = np.array(list(data_generator))
    if sample_from_all_inputs:
        x_train = np.array([random.choice(x_train) for _ in range(no_of_samples)])
    else:
        np.random.shuffle(x_train)
data_generator = itertools.product([0, 1], repeat=12)
all_data = np.array(list(data_generator))


if use_two_xors:
    all_data = np.array(list(filter(filtering, all_data)))
    x_train = np.array(list(filter(filtering, x_train)))
    y_train = np.array(list(map(f1, x_train)))
    all_values = np.array(list(map(f1, all_data)))
else:
    y_train = np.array(list(map(f, x_train)))
    all_values = np.array(list(map(f, all_data)))

if add_label_input:
    all_data = np.column_stack((all_data, all_values))
    x_train = np.column_stack((x_train, y_train))

print(all_data)
print(all_values)
print(all_data.shape)
print(all_values.shape)

print("Size of x_train: ", x_train.shape)
print("Size of y_train: ", y_train.shape)
print(x_train)
print(y_train)

print("Started training", datetime.datetime.now())
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(all_data, all_values), verbose=0, callbacks=[
    # callback.MyCallback(histogram_freq=1, batch_size=4096, write_graph=False, embeddings_freq=0),
    callback2.MyCallback(all_data, frequency, directory),
    TensorBoard(log_dir=directory, histogram_freq=frequency, batch_size=4096, write_graph=False)
])
# model.fit(x_train, y_train, epochs=1, batch_size=100, validation_data=(all_data, all_values), callbacks=[TensorBoard(histogram_freq=1, batch_size=4096, write_graph=False, embeddings_freq=0)])
print("Finished training", datetime.datetime.now())

print("Started counting results")
correct_results = 0
for sample, value in zip(all_data, all_values):
    print(value, model.predict(np.array([sample])))
    correct_results += (value == (model.predict(np.array([sample])) >= 0.5))
    # correct_results += (value == (model.predict(np.array([sample])) >= 0))
print("Finished counting results")

print(correct_results, len(all_data), float(correct_results)/len(all_data))
