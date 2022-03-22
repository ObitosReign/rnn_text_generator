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

# Create the string lookup layer
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None)
# Create the token invert layer
chars_from_ids = tf.keras.layers.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None)

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

model.built = True
model.load_weights('my_model_weights')

# One step prediction
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
    
        # Create a mask to prevent {UNK} from generating
        skip_ids = self.ids_from_chars(['UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put -inf at bad indexes
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape of the vocab
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        
    
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert input to ids
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        
        # Run the model
        predicted_logits, states = self.model(inputs=input_ids, states=states, return_state=True)
        
        # use the last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        
        # Apply the [UNK] prevention mask
        predicted_logits = predicted_logits + self.prediction_mask
            
        # sample the output logits to generate ids
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=1)
        
        # Convert ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)
        
        return predicted_chars, states

# Instance of the one step
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(['ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)

print(result[0].numpy().decode('utf-8'))