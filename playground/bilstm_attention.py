import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.datasets import imdb
from keras.layers import *
from keras.models import *
from keras import backend as K

n_unique_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=n_unique_words)

maxlen = 200
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(n_unique_words, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences

        super(attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1))

        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


model2 = Sequential()
model2.add(Embedding(n_unique_words, 128, input_length=maxlen))
model2.add(Bidirectional(LSTM(64, return_sequences=True)))
model2.add(attention(return_sequences=True))  # receive 3D and output 3D
model2.add(Dropout(0.5))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

history3d = model2.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=12,
                       validation_data=[x_test, y_test])
print(history3d.history['loss'])
print(history3d.history['accuracy'])

model3 = Sequential()
model3.add(Embedding(n_unique_words, 128, input_length=maxlen))
model3.add(Bidirectional(LSTM(64, return_sequences=True)))
model3.add(attention(return_sequences=False))  # receive 3D and output 3D
model3.add(Dropout(0.5))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.summary()

history2d = model3.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=12,
                       validation_data=[x_test, y_test])
print(history3d.history['loss'])
print(history3d.history['accuracy'])
