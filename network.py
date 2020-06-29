import numpy as np
import multiprocessing

from keras import layers
from keras import models
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from dataloader import dataloader


# Model design
def Net():
    model = models.Sequential()
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
    model.compile(optimizer=Adam(learning_rate=1e-5), loss="mse")
    return model


# Training
def train(X, y, model):
    history = model.fit(X, y, epochs=100, validation_split=0.1)

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    process_num = multiprocessing.cpu_count()//2
    p = multiprocessing.Pool(process_num)
    result = []
    for i in range(process_num):
        result.append(p.apply_async(func=dataloader, args=("GBM_lookback", 2500), kwds={"lookback_type":"floating", "option_type": "call"}))
    p.close()
    p.join()
    features = []
    targets = []
    for res in result:
        feature, target = res.get()
        features.extend(feature)
        targets.extend(target)
    features = np.array(features)
    targets = np.array(targets)
    model = Net()
    train(features, targets, model)