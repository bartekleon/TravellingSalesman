import random
import re
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, Flatten, Dense
from keras.optimizers import Adam

def generate_string(length=15, alphabet='abcd'):
  """Generate a random string of given length over the specified alphabet."""
  return ''.join(random.choices(alphabet, k=length))

def check_regex(string, regex_pattern):
  """Check if the string contains a substring that matches the regex pattern."""
  return bool(re.search(regex_pattern, string))

def generate_string_with_pattern(pattern, length=15, alphabet='abcd'):
  random_part = ''.join(random.choices(alphabet, k=length - len(pattern)))
  insert_position = random.randint(0, len(random_part))
  result = random_part[:insert_position] + pattern + random_part[insert_position:]
  return result

# Parameters
num_strings = 10000
regex_pattern = r'abcda'  # Example regex pattern (change as needed)

# Generating strings and labeling them
strings = []
labels = []

frames: list[tuple[str, bool]] = []

for _ in range(5000):
  s = generate_string_with_pattern("abcda")
  frames.append((s, True))

count = 0
while True:
  s = generate_string()
  label = check_regex(s, regex_pattern)
  if not label:
    frames.append((s, False))
    count += 1
  if count == 5000:
    break

random.shuffle(frames)

for strin, label in frames:
  strings.append(strin)
  labels.append(label)

def one_hot_encode(string, alphabet='abcd'):
  """One-hot encode a given string."""
  encoding = {char: np.eye(len(alphabet))[i] for i, char in enumerate(alphabet)}
  return np.array([encoding[char] for char in string])

encoded_strings = np.array([one_hot_encode(s) for s in strings])

X_train, X_test, y_train, y_test = train_test_split(encoded_strings, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, len(y_train), len(y_test))

EPOCHS = 10

# model_simple = Sequential([
#   Conv1D(filters=1, kernel_size=5, activation='relu', input_shape=(15, 4)),
#   Flatten(),
#   Dense(1, activation='sigmoid')
# ])

# model_simple.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
# history_simple = model_simple.fit(X_train, np.array(y_train), epochs=EPOCHS, validation_data=(X_test, np.array(y_test)))
# print(model_simple.layers[0].get_weights()[0])


# model_simple2 = Sequential([
#   Conv1D(filters=16, kernel_size=5, activation='relu', input_shape=(15, 4)),
#   Conv1D(filters=32, kernel_size=5, activation='relu'),
#   Dropout(0.5),
#   Flatten(),
#   Dense(64, activation='relu'),
#   Dropout(0.5),
#   Dense(1, activation='sigmoid')
# ])

#model_simple2.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
#history_simple = model_simple2.fit(X_train, np.array(y_train), epochs=EPOCHS, validation_data=(X_test, np.array(y_test)))

model_complex = Sequential([
  Conv1D(filters=2, kernel_size=5, activation='relu', input_shape=(15, 4)),
  Conv1D(filters=4, kernel_size=5, activation='relu'),
  Flatten(),
  Dense(1, activation='sigmoid')
])

model_complex.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history_simple = model_complex.fit(X_train, np.array(y_train), epochs=EPOCHS, validation_data=(X_test, np.array(y_test)))
