import tensorflow as tf
import matplotlib.pyplot as plt
from Layers import NPU, NAU
import sys
import tempfile

tf.keras.backend.set_floatx('float64')

tf.random.set_seed(1)

EPOCHS = 400
BATCH_SIZE = 32
N = 4000
shape = (N, 2)

f_labels = ["x+y", "x*y", "x/y", "x^2"]


def f(x):
    x_trans = tf.transpose(x)
    result = []
    result += [x_trans[0] + x_trans[1]]  # f = x+y
    result += [x_trans[0] * x_trans[1]]  # f = x*y
    result += [x_trans[0] / x_trans[1]]  # f = x/y
    result += [tf.math.pow(x_trans[0], 2)]  # f = x**2
    result = tf.convert_to_tensor(result)
    return tf.transpose(result)


x_train = tf.random.uniform(minval=-10, maxval=10, shape=shape)
y_train = f(x_train)
x_val = tf.random.uniform(minval=-30, maxval=30, shape=shape)
y_val = f(x_val)
x_test = tf.random.uniform(minval=-30, maxval=30, shape=shape)
y_test = f(x_test)

modelNPU = tf.keras.models.Sequential([
    NPU(4, tau=1e-3)
])
modelNPU_NAU = tf.keras.models.Sequential([
    NPU(8, tau=1e-3),
    NAU(4, tau=1e-3)
])

models = [modelNPU, modelNPU_NAU]

for model in models:
    tmp = tempfile.NamedTemporaryFile(delete=True)
    checkpoint_filepath = tmp.name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    loss_fn = tf.keras.losses.MeanSquaredError()
    model.compile(optimizer='adam',
                  loss=loss_fn)
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint_callback])  # , batch_size=BATCH_SIZE
    model.load_weights(checkpoint_filepath)

    predictions = model.predict(x_test)

    abs_error = tf.abs(y_test - predictions)
    abs_error_log = tf.math.log(abs_error)
    abs_error_log10 = tf.math.log(abs_error) / tf.math.log(tf.constant(10, dtype=float))
    abs_mean_error = tf.transpose(tf.losses.mean_absolute_error(tf.transpose(y_test), tf.transpose(predictions)))
    model.evaluate(x_test, y_test, verbose=2)

    print(model.weights)

    print("Absolute Mean Error")
    for i in range(len(f_labels)):
        tf.print(f_labels[i], ": ", abs_mean_error[i], output_stream=sys.stdout)

    # heatmaps
    for i in range(len(f_labels)):
        plt.title(f_labels[i])
        splot = plt.scatter(tf.transpose(x_test)[0], tf.transpose(x_test)[1], cmap='plasma_r',
                            c=tf.transpose(abs_error)[i])
        plt.xlabel("x")
        plt.ylabel("y")
        plt.colorbar(splot, label="|f-t|")
        # plt.savefig("Plots/functions_NPU_NAU" + str(i) + ".pdf")
        plt.show()
