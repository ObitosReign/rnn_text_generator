import tensorflow as tf
from tensorflow import keras
import numpy as np
import os, time

# Download the dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# Read file & decode for py2 compat
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Get the unique characters in the file
vocab = sorted(set(text))
# Split text into tokens
chars = tf.strings.unicode_split(text, input_encoding='UTF-8')

# Create the string lookup layer
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
# Create the token invert layer
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

# Get the ids
ids = ids_from_chars(chars)

# Convert text vector to stream of character indices
ids_dataset = tf.data.Dataset.from_tensor_slices(ids)

# Sequence length
seq_length = 100

# Create sequences
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

# Shift input to align with input & label
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Create the dataset
dataset = sequences.map(split_input_target)
    
# Batch & buffer size
BATCH_SIZE = 64
BUFFER_SIZE = 10000

# Shuffle & pack data in batches
dataset = (
    dataset
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(tf.data.experimental.AUTOTUNE))

# Vocab length, embedding dimension & number of rnn units
vocab_size = len(ids_from_chars.get_vocabulary())
embedding_dim = 256
rnn_units = 1024

# The model embedding, GRU, dense
class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units,
            return_sequences=True,
                                    return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

# Instance of the model
model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

# Attach loss function
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# Optimize the model
model.compile(optimizer='adam', loss=loss)

# Train the model
EPOCHS = 1

model.built = True
model.load_weights('model_weights/my_model_weights')
history = model.fit(dataset, epochs=EPOCHS)
model.save_weights('model_weights/my_model_weights')