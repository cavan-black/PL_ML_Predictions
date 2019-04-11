import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras.utils import np_utils
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.losses import categorical_crossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

encoder = LabelEncoder()
shift_size = 10
scaler = MinMaxScaler()
batch_size = 50
sequence_length = 380
warmup_steps = 200
optimizer = Adam(lr=0.001)
target_names = ['Away Win', 'Draw', 'Home Win']


def plot_comparison(start_idx, length=100, train=True):
    """
    Plot the predicted and true output-signals.

    :param start_idx: Start-index for the time-series.
    :param length: Sequence-length to process and plot.
    :param train: Boolean whether to use training- or test-set.
    """

    if train:
        # Use training-data.
        x = X_train
        y_true = y_train
    else:
        # Use test-data.
        x = X_test
        y_true = y_test

    # End-index for the sequences.
    end_idx = start_idx + length

    # Select the sequences from the given start-index and
    # of the given length.
    x = x[start_idx:end_idx]
    y_true = y_true[start_idx:end_idx]

    # Input-signals for the model.
    x = np.expand_dims(x, axis=0)

    # Use the model to predict the output-signals.
    y_pred = model.predict(x)

    # The output of the model is between 0 and 1.
    # Do an inverse map to get it back to the scale
    # of the original data-set.
    #y_pred_rescaled = y_scaler.inverse_transform(y_pred[0])

    # For each output-signal.
    for signal in range(len(target_names)):
        # Get the output-signal predicted by the model.
        signal_pred = y_pred[:, signal]

        # Get the true output-signal from the data-set.
        signal_true = y_true[:, signal]

        # Make the plotting-canvas bigger.
        plt.figure(figsize=(15, 5))

        # Plot and compare the two signals.
        plt.plot(signal_true, label='true')
        plt.plot(signal_pred, label='pred')

        # Plot grey box for warmup-period.
        p = plt.axvspan(0, warmup_steps, facecolor='black', alpha=0.15)

        # Plot labels etc.
        plt.ylabel(target_names[signal])
        plt.legend()
        plt.show()


def batch_generator(batch_size, sequence_length):
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, 52)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)
        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, 3)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)
        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # This points somewhere into the training-data.
            idx = np.random.randint(train_len - sequence_length)
            # Copy the sequences of data starting at this index.
            #print(X_train)
            #print(X_train[idx: idx + 200])
            x_batch[i] = X_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]

        yield (x_batch, y_batch)


def reverse_labels(data):
    encoded_y2 = np.argmax(dummy_y, axis=1)
    unencoded_labels = encoder.inverse_transform(encoded_y2)
    return unencoded_labels


if __name__ == "__main__":
    df = pd.read_csv('letres.csv', header=0).drop(['Date', 'Season', 'Unnamed: 0'], axis=1).reset_index(drop=True)
    print(df.columns.values)
    train_size= 0.8
    df.iloc[:, 0:8] = scaler.fit_transform(df.iloc[:, 0:8])
    train_len = int(train_size * len(df.index))
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    encoded_y = encoder.fit_transform(y)
    print(encoder.get_params())
    print(y)
    print(encoded_y)
    #y = y.reshape(len(y), 1)
    dummy_y = np_utils.to_categorical(encoded_y)
    print(dummy_y)


    X_train = X[:train_len]

    #print(X_train.shape)
    X_test = X[train_len:]
    y_train = dummy_y[:train_len]
    #print(y_train)
    y_test = dummy_y[train_len:]
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
    model.add(GRU(units=52,
                  return_sequences=True,
                  input_shape=(None, 52,)))
    model.add(Dense(52))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

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
    #scores = model.evaluate(x=np.expand_dims(X_test, axis=0),
    #                        y=np.expand_dims(y_test, axis=0))
    #print("Accuracy: %.2f%%" % (scores[1] * 100))
    X_test = np.expand_dims(X_test, axis=0)
    y_pred = model.predict(X_test)
    with open('pred.txt', 'w') as f:
        for item in y_pred:
            f.write("%s\n" % item)
    plot_comparison(start_idx=0, length=1000, train=True)
