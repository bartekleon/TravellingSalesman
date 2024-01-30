from tensorflow.keras.layers import Dense, Input, Flatten, Activation, Dropout, Reshape, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data

# model = Sequential([
#   Flatten(input_shape=(28, 28)),
#   Dense(512),
#   Dense(256),
#   Dense(128),
#   Dense(64),
#   Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=5)

# test_loss, test_acc = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {test_acc}")

encoder = Sequential([
    Flatten(input_shape = (28, 28)), # 784
    Dropout(0.5), # The Dropout layer randomly sets input units to 0 with a frequency of `rate`
    Dense(256), # output = activation(dot(input, kernel) + bias)
    LeakyReLU(), # default: 0.3.  f(x) = alpha * x if x < 0  ||  f(x) = x if x >= 0
    Dropout(0.5),
    Dense(128),
    LeakyReLU(),
    Dropout(0.5),
    Dense(64),
    LeakyReLU(),
    Dropout(0.5),
    Dense(32),
    LeakyReLU(),
])
decoder = Sequential([
    Dense(64, input_shape = (32,)),
    LeakyReLU(),
    Dropout(0.5),
    Dense(128),
    LeakyReLU(),
    Dropout(0.5),
    Dense(256),
    LeakyReLU(),
    Dropout(0.5),
    Dense(392),
    LeakyReLU(),
    Dropout(0.5),
    Dense(784),
    LeakyReLU(),
    Activation("sigmoid"),
    Reshape((28, 28))
])

img = Input(shape = (28, 28))
latent_vector = encoder(img)
output = decoder(latent_vector)

model = Model(inputs=img, outputs=output)
model.compile("adam", loss="binary_crossentropy")

EPOCHS = 60

for epoch in range(EPOCHS):
    fig, axs = plt.subplots(4, 4)
    rand = x_test[np.random.randint(0, 10000, 16)].reshape((4, 4, 1, 28, 28))
    
    for i in range(4):
        for j in range(4):
            axs[i, j].imshow(model.predict(rand[i, j])[0], cmap = "gray")
            axs[i, j].axis("off")
    
    plt.subplots_adjust(wspace = 0, hspace = 0)
    plt.show()
    print("-----------", "EPOCH", epoch, "-----------")
    model.fit(x_train, x_train)