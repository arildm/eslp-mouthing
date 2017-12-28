from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Reshape, Flatten, Lambda
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras import backend as K
import numpy as np

import mouth_data

def create_model():
    """Creates an RNN model for sequential frame data input and label output."""
    model = Sequential()
    # Flatten 3D to 1D but keep the bigram dimension.
    #model.add(Reshape((156, 7 * 7 * 2048), input_shape=(156, 7, 7, 2048)))
    model.add(TimeDistributed(Lambda(lambda x: x[:,2:5,0:3]), input_shape=(156, 7, 7, 2048)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(32, return_sequences=True))
    # Output layer the size of the number of labels.
    model.add(TimeDistributed(Dense(40)))
    model.add(Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.summary()
    return model

def get_data():
    """Prepares input and output data for training."""
    data = mouth_data.MouthData()

    # Load frames data.
    frames_data = data.frames_resnet()
    x = data.by_sentence(frames_data)

    # Load annotations.
    print('Loading annotations')
    mouth_annotations = np.array(data.annotation_vectors())
    y = data.by_sentence(mouth_annotations)

    return x, y

def train_evaluate_model():
    """Trains and saves a NN model."""
    # Load data.
    x, y = get_data()

    # Train model.
    print('Creating NN model')
    model = create_model()
    print('Begin training')
    model.fit(x, y, epochs=100, batch_size= 4, callbacks=[
        ModelCheckpoint('mouthing-model-{epoch:02d}.hdf5', 'loss', save_best_only=True),
        EarlyStopping('loss', patience=5),
    ])

if __name__ == '__main__':
    train_evaluate_model()
