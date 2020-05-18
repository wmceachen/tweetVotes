import pandas as pd
import tensorflow.keras as ks
import tensorflow as tf
import gensim.downloader as api
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pickle
from conv_model import conv_model
from lstm_model import lstm_model
from tf_model import tf_model
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from imblearn.over_sampling import SMOTE
from utils import embed_matrix_gen
"""
1) take OG Data
2) Get labels from each model
3) zip, OLS that bad boi
"""
sm = SMOTE(random_state=42)

df = pd.read_csv('sent/trinary_tweets.csv')
#Remove neutral and re encode negative scores
df = df[df.sent != 0].replace({-1: 0})

y = df.sent
X = df.tweet

MAX_SEQ_LEN = 55
VOCAB_SIZE = 20000

vectorizer = TextVectorization(
    max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SEQ_LEN)
text_ds = tf.data.Dataset.from_tensor_slices(X).batch(128)
vectorizer.adapt(text_ds)
X_vecs = vectorizer(np.array([[s] for s in X])).numpy()

#Resample for an even distribution
res_X, res_y = sm.fit_resample(X_vecs, y)

X_train, X_test, y_train, y_test = train_test_split(
    res_X, res_y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42)

#Embedding Stuff
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(2, len(vocab))))
num_tokens = len(vocab) + 2
embedding_dim = 300
embedding_matrix = embed_matrix_gen(word_index, num_tokens, embedding_dim)



conv_model = conv_model(embedding_matrix, MAX_SEQ_LEN, num_tokens, embedding_dim)
lstm_model = lstm_model(embedding_matrix, MAX_SEQ_LEN, num_tokens, embedding_dim)
# tf_model = tf_model(MAX_SEQ_LEN, num_tokens)
# models = [conv_model, lstm_model, tf_model]
for model_str in ['conv_model', 'lstm_model']:
    model = eval(model_str)
    model_callbacks = [
        ks.callbacks.EarlyStopping(
            monitor='val_acc', patience=3, restore_best_weights=True),
        ks.callbacks.ModelCheckpoint(
            filepath=f'models/{model_str}.h5', monitor='val_acc', save_best_only=True)
    ]
    model.summary()
    model.compile("adam", "binary_crossentropy", metrics=['acc'])
    model.fit(X_train, y_train, batch_size=32,
              epochs=20, validation_data=(X_val, y_val), callbacks=model_callbacks)

with open('vectorizer.pickle', 'wb') as handle:
    pickle.dump(vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
# tokenizer = pickle.load(open('vectorizer.pickle', 'rb'))
# tokenizer.transform(X)
