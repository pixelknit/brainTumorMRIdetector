import tensorflow as tf
import pickle
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

dense_layers = [2]
layer_sizes = [128]
conv_layers = [3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            NAME = f"{conv_layer}-{layer_size}-{dense_layer}-{int(time.time())}"
            print(NAME)

            model = Sequential()

            model.add(Conv2D(layer_size,(3,3),input_shape=X.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2,2)))

            for i in range(conv_layer-1):
                model.add(Conv2D(layer_size,(3,3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            model.fit(X, y, batch_size=5, validation_split=0.1,epochs=40, callbacks=[tensorboard])

model.save("brain.model")
