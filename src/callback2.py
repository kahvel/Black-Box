from keras.callbacks import Callback
from keras.models import Model
from keras.layers.core import Activation, Dense

import numpy as np
import pandas as pd


class MyCallback(Callback):
    def __init__(self, function, all_data, frequency):
        super(Callback, self).__init__()
        self.data = {}
        self.function = function
        self.frequency = frequency
        self.all_data = all_data
        self.widths = [12, 10, 7, 5, 4, 3, 2, 1]
        # self.column_names = [["L" + str(i).rjust(2, "0") + "N" + str(j).rjust(2, "0") for j in range(self.widths[i])] for i in range(len(self.widths))]
        # self.flat_column_names = [item for sublist in self.column_names for item in sublist]

    def on_epoch_end(self, epoch, logs=None):
        # model_features = Model(model.input, model.layers[-5].output)
        # model_features.predict()
        if epoch % self.frequency == 0:
            epoch_data = {}
            # print(self.model.layers)
            layer_counter = 0
            for layer in self.model.layers:
                if isinstance(layer, Activation):
                    layer_data = pd.DataFrame(columns=range(self.widths[layer_counter]))
                    model = Model(self.model.input, layer.output)
                    model_output = np.transpose(model.predict(np.array(self.all_data)))
                    # print(layer_data)
                    for i, neuron_data in enumerate(model_output):
                        layer_data.loc[:, i] = neuron_data
                    epoch_data[layer_counter] = layer_data
                    layer_counter += 1
                    # print(layer_data.shape)
                    # if layer_counter == 0:
                    #     import pdb
                    #     pdb.set_trace()
            self.data[epoch] = epoch_data

    def on_train_end(self, logs=None):
        for epoch in self.data:
            writer = pd.ExcelWriter("test" + str(epoch) + ".xlsx", engine="xlsxwriter")
            for layer in self.data[epoch]:
                self.data[epoch][layer].to_excel(writer, sheet_name=str(layer))
