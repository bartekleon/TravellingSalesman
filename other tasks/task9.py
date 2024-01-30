from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import tensorflow as tf

def _normalize_img(img, label):
  img = tf.cast(img, tf.float32) / 255.
  return (img, label)

train_dataset, test_dataset = tfds.load(name="cifar10", split=['train', 'test'], as_supervised=True)

# Build your input pipelines
train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.map(_normalize_img)

test_dataset = test_dataset.batch(32)
test_dataset = test_dataset.map(_normalize_img)


def create_discriminator_model():
  input = Input(shape=(32, 32, 3))

  # Convolutional layers
  x = Conv2D(32, (2, 2), activation='relu')(input)
  x = Conv2D(64, (2, 2), activation='relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(64, (2, 2), activation='relu')(x)
  x = Conv2D(64, (2, 2), activation='relu')(x)
  x = MaxPooling2D()(x)
  x = Conv2D(100, (2, 2), activation='relu')(x)

  # Flatten and output
  x = Flatten()(x)
  x = Dense(200, activation='relu')(x)
  x = Dense(100, activation='relu')(x)
  output = Dense(10, activation='relu')(x)

  return Model(inputs=input, outputs=output)

discriminator = create_discriminator_model()

discriminator.compile(optimizer=Adam(), loss=tfa.losses.TripletSemiHardLoss(), metrics=['accuracy'])
history = discriminator.fit(train_dataset, epochs=5)


discriminator.save_weights('discriminator_weights.h5')

