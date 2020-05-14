import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
# print(tf.test.is_gpu_available())
df = pd.read_csv('sent/trinary_tweets.csv')
df.sent += 1
# df.sent.astype(int)
# y = ks.utils.to_categorical(np.array(df.sent))
X_train, X_test, y_train, y_test = train_test_split(
    df.tweet, np.array(df.sent), test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=1)

MAX_SEQ_LEN = 150
VOCAB_SIZE = 20000

vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LEN)
text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
vectorizer.adapt(text_ds)
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(2, len(vocab))))
wv = api.load('word2vec-google-news-300')
num_tokens = len(vocab) + 2
embedding_dim = 300
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    try:
        embedding_vector = wv.get_vector(word.decode("utf-8"))
        embedding_matrix[i] = embedding_vector
        hits += 1
    except:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

print(embedding_matrix)
X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
X_val = vectorizer(np.array([[s] for s in X_val])).numpy()
X_test = vectorizer(np.array([[s] for s in X_test])).numpy()

tweet_input = ks.layers.Input(shape=(MAX_SEQ_LEN,), dtype='int32')

tweet_encoder = ks.layers.Embedding(num_tokens, embedding_dim, embeddings_initializer=ks.initializers.Constant(
    embedding_matrix), input_length=MAX_SEQ_LEN, trainable=True)(tweet_input)
l1l2_reg = ks.regularizers.l1_l2(l1=0.01, l2=0.01)

merged = []
for n_gram in range(1, 6):
    gram_branch = ks.layers.Conv1D(filters=100, kernel_size=n_gram,
                                   padding='valid', activation='relu', strides=1, kernel_regularizer=l1l2_reg)(tweet_encoder)
    gram_branch = ks.layers.GlobalMaxPooling1D()(gram_branch)
    merged.append(gram_branch)

merged = ks.layers.concatenate(merged, axis=1)

merged = ks.layers.Dense(256, activation='relu', kernel_regularizer=l1l2_reg)(merged)
merged = ks.layers.Dropout(0.2)(merged)
preds = ks.layers.Dense(3, activation="softmax")(merged)
model = ks.models.Model(inputs=[tweet_input], outputs=[preds])
print(model.summary())
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"]
)
my_callbacks = [
    ks.callbacks.EarlyStopping(monitor='val_acc', patience=5,restore_best_weights=True),
    ks.callbacks.ModelCheckpoint(
        filepath='model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_acc', save_best_only=True),
    ks.callbacks.TensorBoard(log_dir='./logs')
]

# # model.load_weights('model.04-1.23.h5')

model.fit(X_train, y_train, batch_size=128,
          epochs=20, validation_data=(X_val, y_val), callbacks=my_callbacks)
# print(accuracy_score(y_test, model.predict(X_test)))
