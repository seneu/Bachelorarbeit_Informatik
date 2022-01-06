import tensorflow as tf
import numpy as np


class Nac(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Nac, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.W_hat = self.add_weight("W_hat",
                                     shape=[int(input_shape[-1]),
                                            self.num_outputs],
                                     initializer='truncated_normal',
                                     trainable=True)
        self.M_hat = self.add_weight("M_hat",
                                     shape=[int(input_shape[-1]),
                                            self.num_outputs],
                                     initializer='truncated_normal',
                                     trainable=True)

    def call(self, inputs):
        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        return tf.matmul(inputs, W)


class Nalu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Nalu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_hat = self.add_weight("W_hat",
                                     shape=shape,
                                     initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02),
                                     trainable=True)
        self.M_hat = self.add_weight("M_hat",
                                     shape=shape,
                                     initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02),
                                     trainable=True)
        self.G = self.add_weight("G",
                                 shape=shape,
                                 initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02),
                                 trainable=True)

    def call(self, inputs):
        epsilon = 1e-7

        W = tf.tanh(self.W_hat) * tf.sigmoid(self.M_hat)
        m = tf.exp(tf.matmul(tf.math.log(tf.abs(inputs) + epsilon), W))
        g = tf.sigmoid(tf.matmul(inputs, self.G))
        a = tf.matmul(inputs, W)

        return g * a + (1 - g) * m


class NaiveNPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(NaiveNPU, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_i = self.add_weight("W_i",
                                   shape=shape,
                                   initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02),
                                   trainable=True)
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.02),
                                   trainable=True)

    def call(self, inputs):
        epsilon = 1e-7
        r = tf.abs(inputs) + epsilon
        k = tf.sign(-(tf.sign(inputs) - 0.5))
        y = tf.exp(tf.matmul(tf.math.log(r), self.W_r) - np.pi * tf.matmul(k, self.W_i)) * tf.math.cos(
            tf.matmul(tf.math.log(r), self.W_i) + np.pi * tf.matmul(k, self.W_r))
        return y


class RealNPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau = 1e-4):
        super(RealNPU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.g = self.add_weight("g",
                                 shape=[1, int(input_shape[-1])],
                                 initializer=tf.keras.initializers.constant(0.5),
                                 trainable=True)

    def _regularizer(self, weight_matrix):
        return self.tau * tf.math.reduce_sum(tf.math.abs(weight_matrix))

    def call(self, inputs):
        epsilon = 1e-7
        #g_hat = tf.minimum(tf.maximum(self.g, 0), 1)
        g_hat = tf.sigmoid(self.g)
        r = g_hat * (tf.abs(inputs) + epsilon) + (1 - g_hat)
        k = tf.sign(-(tf.sign(inputs) - 0.5)) * g_hat
        y = tf.exp(tf.matmul(tf.math.log(r), self.W_r)) * tf.math.cos(np.pi * tf.matmul(k, self.W_r))
        return y


class NPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=1e-4):
        super(NPU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_i = self.add_weight("W_i",
                                   shape=shape,
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.g = self.add_weight("g",
                                 shape=[1, int(input_shape[-1])],
                                 initializer=tf.keras.initializers.constant(0.5),
                                 trainable=True)

    def _regularizer(self, weight_matrix):
        return self.tau * tf.math.reduce_sum(tf.math.abs(weight_matrix))

    def call(self, inputs):
        epsilon = 1e-5
        g_hat = tf.minimum(tf.maximum(self.g, 0), 1)
        r = g_hat * (tf.abs(inputs) + epsilon) + (1 - g_hat)
        k = tf.maximum(-tf.sign(inputs), 0) * g_hat
        y = tf.exp(tf.matmul(tf.math.log(r), self.W_r) - np.pi * tf.matmul(k, self.W_i)) * tf.math.cos(
            tf.matmul(tf.math.log(r), self.W_i) + np.pi * tf.matmul(k, self.W_r))
        return y



class ExponentialNPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=1e-4):
        super(ExponentialNPU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau
        self.Layer_name = "ExponentialNPU"

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_i = self.add_weight("W_i",
                                   shape=shape,
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=self._regularizer,
                                   trainable=True)

        self.phi = self.add_weight("phi",
                                   shape=[1, int(input_shape[-1])],
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   trainable=True)

    def _regularizer(self, weight_matrix):
        return self.tau * tf.reduce_mean(tf.math.abs(weight_matrix))

    def call(self, inputs):
        x = inputs + self.phi
        y = tf.exp(tf.matmul(x, self.W_r)) * tf.math.cos(tf.matmul(x, self.W_i))
        return y

class NAU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=0.01):
        super(NAU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.A = self.add_weight("A",
                                 shape=shape,
                                 initializer=tf.keras.initializers.Zeros(),
                                 regularizer=self._regularizer,
                                 trainable=True)

    def _regularizer(self, weight_matrix):
        weight_matrix_clamped = tf.minimum(tf.maximum(weight_matrix, -1), 1)
        return self.tau * tf.math.reduce_mean(
            tf.math.minimum(tf.math.abs(weight_matrix_clamped), 1 - tf.math.abs(weight_matrix_clamped)))

    def call(self, inputs):
        A_hat = tf.minimum(tf.maximum(self.A, -1), 1)
        y = tf.matmul(inputs, A_hat)
        return y


class NMU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=0.01):
        super(NMU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau

    def build(self, input_shape):
        self.shape = [int(input_shape[-1]), self.num_outputs]
        self.M = self.add_weight("M",
                                 shape=self.shape,
                                 initializer=tf.keras.initializers.Zeros(),
                                 regularizer=self._regularizer,
                                 trainable=True)

    def _regularizer(self, weight_matrix):
        weight_matrix_clamped = tf.minimum(tf.maximum(weight_matrix, 0), 1)
        return self.tau * tf.math.reduce_mean(
            tf.math.minimum(weight_matrix_clamped, 1 - weight_matrix_clamped))

    def call(self, inputs):
        M_hat = tf.minimum(tf.maximum(self.M, 0), 1)
        y = []
        for i in range(M_hat.shape[1]):
            M_hat_t = tf.transpose(M_hat)
            y += [tf.transpose(tf.math.reduce_prod(M_hat_t[i] * inputs + 1 - M_hat_t[i], 1))]
        y = tf.convert_to_tensor(y)
        return tf.transpose(y)


class LogLogNPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=1e-4):
        super(LogLogNPU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau
        self.Layer_name = "LogLogNPU"

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_i = self.add_weight("W_i",
                                   shape=shape,
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.g = self.add_weight("g",
                                 shape=[1, int(input_shape[-1])],
                                 initializer=tf.keras.initializers.constant(0.5),
                                 trainable=True)
        self.phi = self.add_weight("phi",
                                   shape=[1, int(input_shape[-1])],
                                   initializer=tf.keras.initializers.Zeros(),
                                   trainable=True)

    def _regularizer(self, weight_matrix):
        return self.tau * tf.reduce_mean(tf.math.abs(weight_matrix))
        # return self.tau * tf.math.reduce_sum(tf.math.abs(weight_matrix))

    def call(self, inputs):
        epsilon = 1e-4
        x = inputs

        g_hat = tf.minimum(tf.maximum(self.g, 0), 1)
        # x = g_hat*x+(1-g_hat)
        r = g_hat * (tf.abs(x) + epsilon) + (1 - g_hat)  # tf.abs(x) + epsilon
        # r = tf.abs(x) + epsilon
        k = tf.maximum(-tf.sign(x), 0) * g_hat

        phi_z = tf.math.atan2(np.pi * k + self.phi, tf.math.log(r))
        z_abs = tf.sqrt(tf.math.log(r) ** 2 + (np.pi * k + self.phi) ** 2)

        u = tf.matmul(tf.math.log(z_abs + epsilon), self.W_r) - tf.matmul(phi_z, self.W_i)
        v = tf.matmul(tf.math.log(z_abs + epsilon), self.W_i) + tf.matmul(phi_z, self.W_r)

        y = tf.exp(tf.exp(u) * tf.math.cos(v)) * tf.math.cos(tf.exp(u) * tf.math.sin(v))
        return y


class LogNPU(tf.keras.layers.Layer):
    def __init__(self, num_outputs, tau=1e-4):
        super(LogNPU, self).__init__()
        self.num_outputs = num_outputs
        self.tau = tau
        self.Layer_name = "LogNPU"

    def build(self, input_shape):
        shape = [int(input_shape[-1]), self.num_outputs]
        self.W_i = self.add_weight("W_i",
                                   shape=shape,
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.W_r = self.add_weight("W_r",
                                   shape=shape,
                                   initializer=tf.keras.initializers.GlorotUniform(),
                                   regularizer=self._regularizer,
                                   trainable=True)
        self.g = self.add_weight("g",
                                 shape=[1, int(input_shape[-1])],
                                 initializer=tf.keras.initializers.constant(0.5),
                                 regularizer=self._regularizer,
                                 trainable=True)
        self.phi = self.add_weight("phi",
                                   shape=[1, int(input_shape[-1])],
                                   initializer=tf.keras.initializers.Zeros(),
                                   regularizer=self._regularizer,
                                   trainable=True)

    def _regularizer(self, weight_matrix):
        return self.tau * tf.reduce_mean(tf.math.abs(weight_matrix))

    def call(self, inputs):
        epsilon = 1e-4

        x = inputs + self.phi
        g_hat = tf.minimum(tf.maximum(self.g, 0), 1)
        r = g_hat * (tf.abs(x) + epsilon) + (1 - g_hat)

        k = tf.maximum(-tf.sign(x), 0) * g_hat

        u = tf.exp(tf.matmul(tf.math.log(r), self.W_r) - np.pi * tf.matmul(k, self.W_i))
        v = np.pi * tf.matmul(k, self.W_r) + tf.matmul(tf.math.log(r), self.W_i)

        y = tf.exp(u * tf.math.cos(v)) * tf.math.cos(u * tf.math.sin(v))
        return y
