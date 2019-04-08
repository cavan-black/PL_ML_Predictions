import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

shift_size = 10
scaler = MinMaxScaler()
batch_size = 64
sequence_length = 200
warmup_steps = 50
optimizer = Adam(lr=0.001)


def batch_generator(batch_size, sequence_length):
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, 48)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, 1)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(train_len - sequence_length)
            # Copy the sequences of data starting at this index.
            #print(X_train)
            #print(X_train[idx: idx + 200])
            x_batch[i] = X_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]

        yield (x_batch, y_batch)


def loss_mse_warmup(y_true, y_pred):
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """

    # The shape of both input tensors are:
    # [batch_size, sequence_length, num_y_signals].

    # Ignore the "warmup" parts of the sequences
    # by taking slices of the tensors.
    y_true_slice = y_true[:, warmup_steps:, :]
    y_pred_slice = y_pred[:, warmup_steps:, :]

    # These sliced tensors both have this shape:
    # [batch_size, sequence_length - warmup_steps, num_y_signals]

    # Calculate the MSE loss for each value in these tensors.
    # This outputs a 3-rank tensor of the same shape.
    loss = tf.losses.mean_squared_error(labels=y_true_slice,
                                        predictions=y_pred_slice)

    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire tensor, we reduce it to a
    # single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)

    return loss_mean


if __name__ == "__main__":
    df = pd.read_csv('wpl.csv', header=0).reset_index(drop=True)

    train_size= 0.8
    df.iloc[:, 0:8] = scaler.fit_transform(df.iloc[:, 0:8])
    train_len = int(train_size * len(df.index))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    y = y.reshape(len(y), 1)

    print(y.shape)
    X_train = X[:train_len]
    #print(X_train.shape)
    X_test = X[train_len:]
    y_train = y[:train_len]
    y_test = y[train_len:]
    generator = batch_generator(batch_size=batch_size, sequence_length=sequence_length)
    x_batch, y_batch = next(generator)
    batch = 0  # First sequence in the batch.
    signal = 0  # First signal from the 20 input-signals.
    #seq = x_batch[batch, :, signal]
    #plt.plot(seq)
    #plt.show()
    validation_data = (np.expand_dims(X_test, axis=0),
                       np.expand_dims(y_test, axis=0))
    model = Sequential()
    model.add(GRU(units=512,
                  return_sequences=True,
                  input_shape=(None, 48,)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_mse_warmup, optimizer=optimizer)
    print(model.summary())
    path_checkpoint = 'checkpoint.keras'
    callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                          monitor='val_loss',
                                          verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True)
    callback_early_stopping = EarlyStopping(monitor='val_loss',
                                            patience=5, verbose=1)
    callback_tensorboard = TensorBoard(log_dir='./logs/',
                                       histogram_freq=0,
                                       write_graph=False)
    callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.1,
                                           min_lr=1e-4,
                                           patience=0,
                                           verbose=1)
    callbacks = [callback_early_stopping,
                 callback_checkpoint,
                 callback_tensorboard,
                 callback_reduce_lr]
    model.fit_generator(generator=generator,
                        epochs=20,
                        steps_per_epoch=100,
                        validation_data=validation_data,
                        callbacks=callbacks)
    try:
        model.load_weights(path_checkpoint)
    except Exception as error:
        print("Error trying to load checkpoint.")
        print(error)
    result = model.evaluate(x=np.expand_dims(X_test, axis=0),
                            y=np.expand_dims(y_test, axis=0))

    print("loss (test-set):", result)
