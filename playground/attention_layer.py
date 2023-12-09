from tensorflow import keras
from keras import layers

# https://analyticsindiamag.com/a-beginners-guide-to-using-attention-layer-in-neural-networks/

query = keras.Input(shape=(None,), dtype='int32')
value = keras.Input(shape=(None,), dtype='int32')

token_embedding = layers.Embedding(input_dim=1000, output_dim=64)

query_embeddings = token_embedding(query)
value_embeddings = token_embedding(value)

layer_cnn = layers.Conv1D(filters=100, kernel_size=4, padding='same')

query_encoding = layer_cnn(query_embeddings)
value_encoding = layer_cnn(value_embeddings)

query_attention_seq = layers.Attention()([query_encoding, value_encoding])

query_encoding = layers.GlobalAveragePooling1D()(query_encoding)
query_value_attention = layers.GlobalAveragePooling1D()(query_attention_seq)

input_layer = layers.Concatenate()([query_encoding, query_value_attention])
