from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

(train_images, train_labels) = mnist.load_data()[0]
(test_images, test_labels) = mnist.load_data()[1]

train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], 1)

train_images = train_images.astype("float32")
test_images = test_images.astype("float32")

train_images = train_images / 255
test_images = test_images / 255

input_shape = (28, 28, 1)

model = Sequential()

model.add(Conv2D(28, kernel_size=(3, 3), input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x = train_images, y = train_labels, epochs=10)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Loss", test_loss)
print("Test accuracy", test_accuracy)

history_dict = history.history
print("Keys : ", history_dict.keys())

epochs = range(1, 11)
loss = history_dict["loss"]
accuracy = history_dict["accuracy"]

plt.plot(epochs, loss)
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(epochs, accuracy)
plt.title("accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.show()

model.save("mnist_model.h5")