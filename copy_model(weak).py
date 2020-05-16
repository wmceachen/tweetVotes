import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
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
y = np.array(df.sent)
# y = np.array(df.sent).reshape(len(df.sent), 1)
# enc_y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    df.tweet, y, test_size=0.2, random_state=1)

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
embedding_dim = 300


def word2vec_matrix_gen(word_index, vocab_size, embedding_dim):
    # Prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    wv = api.load('word2vec-google-news-300')
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
embedding_layer = ks.layers.Embedding(
    num_tokens,
    embedding_dim,
    embeddings_initializer=ks.initializers.Constant(embedding_matrix),
    trainable=False,
)
int_sequences_input = ks.layers.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = ks.layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = ks.layers.MaxPooling1D(5)(x)
x = ks.layers.Conv1D(128, 5, activation="relu")(x)
x = ks.layers.MaxPooling1D(5)(x)
x = ks.layers.Conv1D(128, 5, activation="relu")(x)
x = ks.layers.GlobalMaxPooling1D()(x)
x = ks.layers.Dense(128, activation="relu")(x)
x = ks.layers.Dropout(0.5)(x)
preds = ks.layers.Dense(3, activation="softmax")(x)
model = ks.Model(int_sequences_input, preds)
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
model.fit(X_train, y_train, batch_size=128, epochs=20, validation_data=(X_val, y_val))
