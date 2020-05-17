import pandas as pd
import tensorflow.keras as ks
from tensorflow.keras import layers
import tensorflow as tf
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pickle

# print(tf.test.is_gpu_available())


class MultiHeadSelfAttention(ks.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(ks.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = ks.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(ks.layers.Layer):
    def __init__(self, maxlen, vocab_size, emded_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(
            input_dim=vocab_size, output_dim=emded_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=emded_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def tf_model(vocab_size, max_seq_len, embed_dim=32, num_heads=2, ff_dim=32):
    # embed_dim: Embedding size for each token
    # num_heads: Number of attention heads
    # ff_dim: Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_seq_len,))
    embedding_layer = TokenAndPositionEmbedding(
        max_seq_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = ks.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 55  # Only consider the first 200 words of each movie review

    df = pd.read_csv('sent/trinary_tweets.csv')
    df.sent += 1
    y = np.array(df.sent)
    # y = np.array(df.sent).reshape(len(df.sent), 1)
    # enc_y = enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        df.tweet, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1)

    vectorizer = TextVectorization(
        max_tokens=vocab_size, output_sequence_length=maxlen)
    text_ds = tf.data.Dataset.from_tensor_slices(X_train).batch(128)
    vectorizer.adapt(text_ds)
    X_train = vectorizer(np.array([[s] for s in X_train])).numpy()
    X_val = vectorizer(np.array([[s] for s in X_val])).numpy()
    X_test = vectorizer(np.array([[s] for s in X_test])).numpy()

    vocab = vectorizer.get_vocabulary()
    vocab_size = len(vocab) + 2
    my_callbacks = [ks.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True),
        ks.callbacks.ModelCheckpoint(
        filepath='models/tf_model.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_accuracy', save_best_only=True)
    ]
    model.compile("adam", "binary_crossentropy")
    model.fit(X_train, y_train, batch_size=32, epochs=20,
              validation_data=(X_val, y_val), callbacks=my_callbacks)
    pickle.dump(vectorizer, open("models/tf_vector.pickel", "wb"))
