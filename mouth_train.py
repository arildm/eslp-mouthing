from keras.layers.core import Activation, Dense, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import nltk
import numpy as np

import mouth_data

def create_model():
    """Creates an RNN model for sequential frame data input and label output."""
    model = Sequential()
    # Flatten 3D to 1D but keep the bigram dimension.
    model.add(Reshape((2, 7 * 7 * 2048), input_shape=(2, 7, 7, 2048)))
    model.add(LSTM(32))
    # Output layer the size of the number of labels.
    model.add(Dense(40))
    model.add(Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.summary()
    return model

def train_evaluate_model():
    """Trains and saves a model."""
    data = mouth_data.MouthData()

    model = create_model()

    # Load frames data and convert to bigrams.
    frames_data = data.frames_resnet()
    frames_data_bigrams = np.array(list(nltk.bigrams(frames_data)))

    # Load annotations; skip first annotation because input is bigrams.
    mouth_annotations = np.array(data.annotation_vectors()[1:])

    # Train and save model.
    model.fit(frames_data_bigrams, mouth_annotations)
    model.save('mouthing-model.h5')

if __name__ == '__main__':
    train_evaluate_model()
