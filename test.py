from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import load_model
import random

(train_images, train_labels) = mnist.load_data()[0]
(test_images, test_labels) = mnist.load_data()[1]

model = load_model("mnist_model.h5")

i = random.randint(1,5000)

prediction = model.predict(test_images[i].reshape(1, 28, 28, 1))

print("Prediction Number : ", prediction.argmax())

plt.imshow(test_images[i].reshape(28, 28), cmap="gray_r")
plt.show()
