import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
print(keras.__version__)

import torch
print(torch.__version__)


import pandas as pd
import matplotlib.pyplot as plt

# instanciamos el dataset de mnist
mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

# separamos para validar y entrenar
X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.


# instanciamos las capas de la red neuronal
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# para ver un resumen de nuestro modelo
model.summary()

# instanciamos el compile con la función de pérdida, optimizador y métricas
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])


# entrenamos el modelo de red neuronal
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))


# mostramos el historial de entrenamiento
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# calculamos las métricas
model.evaluate(X_test, y_test)

# hacemos prediciones con datos nuevos
X_new = X_test[:3]
y_proba = model.predict(X_new)
y_proba.round(2)


# mostramos las imagenes para ver el resultado del modelo
plt.figure(figsize=(7.2, 2.4))
for index, image in enumerate(X_new):
    plt.subplot(1, 3, index + 1)
    plt.imshow(image, cmap="binary", interpolation="nearest")
    plt.axis('off')
    plt.title(y_test[index], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

