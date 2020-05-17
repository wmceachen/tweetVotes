import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from imblearn.over_sampling import SMOTE
import pickle
# print(tf.test.is_gpu_available())
# enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
sm = SMOTE(random_state=42)

df = pd.read_csv('sent/trinary_tweets.csv')
df = df[df.sent != 0].replace({-1: 0})
df = df[(df.tweet.str.split().apply(len) > 4) &
        (df.tweet.str.split().apply(len) < 50)]
y = df.sent
X = df.tweet
# # y = ks.utils.to_categorical(np.array(df.sent))
# y = np.array(df.sent).reshape(len(df.sent), 1)
# enc_y = enc.fit_transform(y)

MAX_SEQ_LEN = 55
VOCAB_SIZE = 20000

vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LEN)
text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
vectorizer.adapt(text_ds)
X_vecs = vectorizer(np.array([[s] for s in X])).numpy()
res_X, res_y = sm.fit_resample(X_vecs, y)

X_train, X_test, y_train, y_test = train_test_split(
    res_X, res_y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)


vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(2, len(vocab))))
num_tokens = len(vocab) + 2
embedding_dim = 300


def word2vec_matrix_gen(word_index, vocab_size, embedding_dim):
    # Prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    # wv = api.load('glove-twitter-100')
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


def conv_model(embedding_matrix, text_len, vocab_size):
    # Given an embedding matrix, return
    tweet_input = ks.layers.Input(shape=(text_len,), dtype='int32')

    tweet_encoder = ks.layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=ks.initializers.Constant(
        embedding_matrix), input_length=text_len, trainable=False)(tweet_input)
    l1l2_reg = ks.regularizers.l1_l2(l1=0.01, l2=0.01)
    merged = []
    for n_gram in range(1, 5):
        gram_branch = ks.layers.Conv1D(filters=100, kernel_size=n_gram,
                                       padding='valid', activation='relu', strides=1)(tweet_encoder)
        gram_branch = ks.layers.GlobalMaxPooling1D()(gram_branch)
        merged.append(gram_branch)

    merged = ks.layers.concatenate(merged, axis=1)

    merged = ks.layers.Dense(256, activation='relu',
                             kernel_regularizer=l1l2_reg)(merged)
    merged = ks.layers.Dropout(0.2)(merged)
    # preds = ks.layers.Dense(3, activation="softmax")(merged)
    preds = ks.layers.Dense(1, activation="sigmoid")(merged)
    model = ks.models.Model(inputs=[tweet_input], outputs=[preds])
    return model


my_callbacks = [
    ks.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True),
    ks.callbacks.ModelCheckpoint(
        filepath='models/conv_model.{epoch:02d}-{val_acc:.2f}.h5', monitor='val_accuracy', save_best_only=True)
]#,ks.callbacks.TensorBoard(log_dir='./logs')
embedding_matrix = word2vec_matrix_gen(word_index, num_tokens, embedding_dim)
model = conv_model(embedding_matrix, MAX_SEQ_LEN, num_tokens)

model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=['acc']
)
model.fit(X_train, y_train, batch_size=32,
          epochs=20, validation_data=(X_val, y_val), callbacks=my_callbacks)
pickle.dump(vectorizer, open("models/conv_vector.pickel", "wb"))
# print(accuracy_score(y_test, model.predict(X_test)))
