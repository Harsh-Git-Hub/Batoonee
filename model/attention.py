import keras

"""
Author: Harsh Gupta
Created on: 2nd Nov 2020 7:00 PM
"""


class Attention(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = keras.activations.tanh(keras.layers.Dot(inputs, self.W) + self.b)
        a = keras.activations.softmax(x, axis=1)
        # a = keras.backend.expand_dims(a, axis=-1)
        output = x * a
        return output
        # return keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Attention, self).get_config()
