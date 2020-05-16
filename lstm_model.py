import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
from tensorflow.keras import layers
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# print(tf.test.is_gpu_available())
enc = OneHotEncoder(handle_unknown='ignore', sparse=False)

df = pd.read_csv('sent/trinary_tweets.csv')
df.sent += 1

# y = ks.utils.to_categorical(np.array(df.sent))
# y = np.array(df.sent).reshape(len(df.sent), 1)
# enc_y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    df.tweet, df.sent, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

MAX_SEQ_LEN = 150
VOCAB_SIZE = 20000

vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LEN)
text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
vectorizer.adapt(text_ds)
X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
X_val = vectorizer(np.array([[s] for s in X_val])).numpy()
X_test = vectorizer(np.array([[s] for s in X_test])).numpy()

vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(2, len(vocab))))
num_tokens = len(vocab) + 2
embedding_dim = 100


def word2vec_matrix_gen(word_index, vocab_size, embedding_dim):
    # Prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    wv = api.load('glove-twitter-100')
    hits = 0
    misses = 0
    for word, i in word_index.items():
        try:
            embedding_vector = wv.get_vector(word.decode("utf-8"))
            embedding_matrix[i] = embedding_vector
            hits += 1
        except:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix


embedding_matrix = word2vec_matrix_gen(word_index, num_tokens, embedding_dim)
tweet_input = ks.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')
# Embed each integer in a 100-dimensional vector
tweet_encoder = ks.layers.Embedding(num_tokens, embedding_dim, embeddings_initializer=ks.initializers.Constant(
    embedding_matrix), input_length=MAX_SEQ_LEN, trainable=False)(tweet_input)
# Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(tweet_encoder)
x = layers.Bidirectional(layers.LSTM(64))(x)
# Add a classifier
outputs = layers.Dense(1)(x)
model = ks.Model(tweet_input, outputs)
model.summary()
model.compile("adam", "mean_squared_error")
my_callbacks = [ks.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)]
model.fit(X_train, y_train, batch_size=32,
          epochs=20, validation_data=(X_val, y_val), callbacks=my_callbacks)
