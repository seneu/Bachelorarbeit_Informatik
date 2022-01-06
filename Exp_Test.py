import tensorflow as tf
import matplotlib.pyplot as plt
import tikzplotlib
from Layers import LogLogNPU, LogNPU, ExponentialNPU, NPU
import tempfile

save_to_file = False

tf.keras.backend.set_floatx('float64')
tf.random.set_seed(0)
def sort_pairs(x, y):
    zipped_lists = zip(x, y)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    x, y = [list(tuple) for tuple in tuples]
    return x, y

params = {'font.size' : 11
          }
plt.rcParams.update(params)

n = 2000  # Number of Datapoints for Training And Validation
shape = [n, 1]

epochs = 400
learning_rate = 1e-3

f = lambda x: tf.math.exp(x)

x_train = tf.random.uniform(minval=-10, maxval=10, shape=shape)
y_train = f(x_train)
x_val = tf.random.uniform(minval=-30, maxval=30, shape=shape)
y_val = f(x_val)
x_test = tf.random.uniform(minval=-40, maxval=40, shape=shape)
y_test = f(x_test)

tau = 1e-3
model_LogLogNPU = tf.keras.models.Sequential([
    LogLogNPU(1, tau=tau)
])
model_LogNPU = tf.keras.models.Sequential([
    LogNPU(1, tau=tau)
])

model_ExponentialNPU = tf.keras.models.Sequential([
    ExponentialNPU(1, tau=tau)
])

model_NPU =  tf.keras.models.Sequential([
    NPU(1, tau=tau)
])

models = [model_LogLogNPU, model_LogNPU, model_ExponentialNPU, model_NPU]
models_names = ["LogLogNPU", "LogNPU", "ExponentialNPU", "NPU"]

for i in range(len(models)):
    model = models[i]
    name = models_names[i]
    # checkpoint for best model out of EPOCHS
    tmp = tempfile.NamedTemporaryFile(delete=True)
    checkpoint_filepath = tmp.name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    loss_fn = tf.keras.losses.MeanSquaredLogarithmicError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss_fn)
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),
                        callbacks=[model_checkpoint_callback])

    # load model with best results
    model.load_weights(checkpoint_filepath)
    results = model.predict(x_test)
    model.evaluate(x_test, y_test, verbose=2)
    model.summary()

    # plot
    sorted_x_test, sorted_results = sort_pairs(x_test, results)
    plt.semilogy(sorted_x_test, sorted_results)
    sorted_x_test, sorted_y_test = sort_pairs(x_test, y_test)
    plt.semilogy(sorted_x_test, sorted_y_test)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('Model Result')
    plt.legend([name, "Test"])

    if save_to_file:
        tikzplotlib.save("Plots/" + name + "_exp.tex", axis_width=r"\textwidth")
        plt.clf()
    else:
        plt.show()

    # plot loss history
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Test'], loc='upper right')

    if save_to_file:
        tikzplotlib.save("Plots/" + name + "_Loss_exp.tex", axis_width=r"\textwidth")
        plt.clf()
    else:
        plt.show()

