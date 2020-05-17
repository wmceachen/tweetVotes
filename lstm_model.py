import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
from tensorflow.keras import layers
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from imblearn.over_sampling import SMOTE
import pickle

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# print(tf.test.is_gpu_available())


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


def lstm_model(embedding_matrix, text_len, vocab_size):
    l1l2_reg = ks.regularizers.l1_l2(l1=0.01, l2=0.01)
    tweet_input = ks.layers.Input(shape=(text_len,), dtype='int32')

    tweet_encoder = ks.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=ks.initializers.Constant(
        embedding_matrix), input_length=text_len, trainable=False)(tweet_input)
    x = layers.Bidirectional(layers.LSTM(
        64, return_sequences=True))(tweet_encoder)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = ks.Model(tweet_input, outputs)
    return model


if __name__ == "__main__":
    sm = SMOTE(random_state=42)

    df = pd.read_csv('sent/trinary_tweets.csv')
    df = df[df.sent != 0].replace({-1: 0})
    MAX_SEQ_LEN = 55
    VOCAB_SIZE = 20000

    vectorizer = TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LEN)
    text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
    vectorizer.adapt(text_ds)
    X_vecs = vectorizer(np.array([[s] for s in X])).numpy()
    res_X, res_y = sm.fit_resample(X_vecs, y)

    X_train, X_test, y_train, y_test = train_test_split(
        res_X, res_y, test_size=0.15, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    vocab = vectorizer.get_vocabulary()
    word_index = dict(zip(vocab, range(2, len(vocab))))
    num_tokens = len(vocab) + 2
    embedding_dim = 100
    embedding_matrix = word2vec_matrix_gen(
        word_index, num_tokens, embedding_dim)
    model = lstm_model(embedding_matrix, MAX_SEQ_LEN, num_tokens)
    model.summary()
    model.compile("adam", "binary_crossentropy")
    my_callbacks = [ks.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True),
        ks.callbacks.ModelCheckpoint(
            filepath='models/lstm_model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_accuracy', save_best_only=True)]
    model.fit(X_train, y_train, batch_size=32,
              epochs=20, validation_data=(X_val, y_val), callbacks=my_callbacks)
    pickle.dump(vectorizer, open("models/conv_vector.pickel", "wb"))
