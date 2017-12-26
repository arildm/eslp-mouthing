from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Reshape
from keras.layers.recurrent import LSTM
from keras.models import Sequential
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
    """Trains and saves a NN model."""
    # @todo We are ignoring that frames are taken from multiple clips.
    data = mouth_data.MouthData()

    # Load frames data as bigrams.
    frames_data_bigrams = data.frames_resnet_bigrams()

    # Load annotations.
    print('Loading annotations')
    mouth_annotations = np.array(data.annotation_vectors())

    # Train model.
    print('Creating NN model')
    model = create_model()
    print('Begin training')
    model.fit(frames_data_bigrams, mouth_annotations, epochs=100, callbacks=[
        ModelCheckpoint('mouthing-model-{epoch:02d}.hdf5', 'loss', save_best_only=True),
        EarlyStopping('loss', patience=5),
    ])

if __name__ == '__main__':
    train_evaluate_model()
