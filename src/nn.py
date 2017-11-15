import callback, callback2

import numpy as np
import itertools
import datetime

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

# import entropy_estimators
# import pydot
# print(pydot.find_graphviz())
model = Sequential()

model.add(Dense(12, input_dim=12))
model.add(Activation("tanh"))
model.add(Dense(10))
model.add(Activation("tanh"))
model.add(Dense(7))
model.add(Activation("tanh"))
model.add(Dense(5))
model.add(Activation("tanh"))
model.add(Dense(4))
model.add(Activation("tanh"))
model.add(Dense(3))
model.add(Activation("tanh"))
model.add(Dense(2))
model.add(Activation("tanh"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
# model.add(Dense(1))

model.compile(#loss='categorical_crossentropy',
              loss='binary_crossentropy',
              optimizer='sgd',)
              #metrics=['accuracy'])
print("Compiled model")
# plot_model(model, to_file='model.png')


def f(x):
    y = [0,0,1,1,None,None,None,None,None,None,None,None]
    return int(all(map(lambda a: ((a[0] == a[1]) if a[1] is not None else True), zip(x, y))))


x_train = np.array([np.random.random_integers(0, 1, 12) for _ in range(1000)])
y_train = np.array(list(map(f, x_train)))

print("Size of x_train: ", x_train.shape)
print("Size of y_train: ", y_train.shape)
print(x_train)
print(y_train)

# y_train = to_categorical(y_train)
# print(y_train)


data_generator = itertools.product([0, 1], repeat=12)
all_data = np.array(list(data_generator))
all_values = np.array(list(map(f, all_data)))
print(all_data)
print(all_values)
print(all_data.shape)
print(all_values.shape)

print("Started training", datetime.datetime.now())
model.fit(x_train, y_train, epochs=10000, batch_size=100, validation_data=(all_data, all_values), callbacks=[
    # callback.MyCallback(histogram_freq=1, batch_size=4096, write_graph=False, embeddings_freq=0),
    callback2.MyCallback(f, all_data, 1000)
])
# model.fit(x_train, y_train, epochs=1, batch_size=100, validation_data=(all_data, all_values), callbacks=[TensorBoard(histogram_freq=1, batch_size=4096, write_graph=False, embeddings_freq=0)])
print("Finished training", datetime.datetime.now())

print("Started counting results")
correct_results = 0
for sample in all_data:
    # print(f(sample), model.predict(np.array([sample])))
    correct_results += (f(sample) == (model.predict(np.array([sample])) >= 0.5))
print("Finished counting results")

print(correct_results, 2**12, float(correct_results)/(2**12))
