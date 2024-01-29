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


def create_classification_model(trained_discriminator, freeze_layers=True):
  if freeze_layers:
    for layer in trained_discriminator.layers:
      layer.trainable = False

  # Add fully-connected layers for classification
  x = Dense(128, activation='relu')(trained_discriminator.output)
  output = Dense(10, activation='softmax')(x)

  return Model(inputs=trained_discriminator.input, outputs=output)

# Create three different models for the three scenarios
# 1. Randomly initialized model
model_random = create_classification_model(create_discriminator_model(), freeze_layers=False)

# 2. Model with weights for conv layers loaded from trained discriminator
discriminator.load_weights('discriminator_weights.h5')
model_pretrained = create_classification_model(discriminator, freeze_layers=False)

# 3. Model with weights for conv layers loaded from trained discriminator and frozen
model_pretrained_frozen = create_classification_model(discriminator, freeze_layers=True)

model_random.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
