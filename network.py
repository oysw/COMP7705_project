import numpy as np

from keras import layers
from keras import models
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataloader import dataloader

# Model design

def Net():
    model = models.Sequential()
#     model.add(layers.Dense(1024, input_shape=(d_input,)))
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.ELU())
    model.add(layers.Dense(1024))
    model.add(layers.ELU())
    model.add(layers.Dense(1024))
    model.add(layers.ELU())
    model.add(layers.Dense(1024))
    model.add(layers.ELU())
    model.add(layers.Dense(1))
    model.compile(Adam(learning_rate=1e-6), "mse", ["acc"])
    return model

# Training

def train(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # features, target = dataloader("GBM_barrier", "call", 1000, knock_type="out", barrier_type="down", barrier_price=90)
    features, target = dataloader("GBMSA_AM", "call", sample_size=100)
    features = np.array(features)
    target = np.array(target)
    model = Net()
    train(features, target, model)