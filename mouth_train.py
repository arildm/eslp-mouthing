from itertools import cycle, repeat

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential, load_model
import mouth_data
import numpy as np
import sys

def chunk(gen, size):
    """Yields a list of size elements from gen at a time."""
    buf = []
    for x in gen:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    yield buf

def chunk_pad(gen, chunk_size, in_sizes, paditem):
    """Splits gen into chunks at indicated points & pads them to equal size."""
    buf = []
    i = 0
    for x in gen:
        buf.append(x)
        if len(buf) >= in_sizes[i]:
            buf += repeat(paditem, chunk_size - len(buf))
        if len(buf) >= chunk_size:
            yield buf
            buf = []
            i += 1

def create_model():
    """Creates an RNN model for sequential frame data input and label output."""
    model = Sequential()
    # Flatten 3D to 1D but keep the bigram dimension.
    # Division of work: Mehdi came up with and implemented Keras-level cropping.
    # model.add(TimeDistributed(Lambda(lambda x: x[:,2:5,0:3]), input_shape=(156, 7, 7, 2048)))
    # model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Flatten(), input_shape=(156, 7, 7, 2048)))
    # LSTM helps with sequential data.
    model.add(LSTM(32, return_sequences=True))
    # Output layer the size of the number of labels.
    model.add(TimeDistributed(Dense(40)))
    # Softmax and categorical cross entropy are good for classification.
    model.add(Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.summary()
    return model

def get_data(mouth_data, batch_size):
    """Generates sentence-wise input & output data batches, indefinitely."""
    xy_gen = zip(mouth_data.frames_resnet(), mouth_data.annotation_vectors())
    in_sizes = [len(mouth_data.paths_by_sentence[sentid])
        for sentid in mouth_data.sentences]
    z = (np.zeros((7, 7, 2048)), np.zeros((40)))
    xy_sentwise = chunk_pad(xy_gen, 156, in_sizes, z)
    for sents in chunk(cycle(xy_sentwise), batch_size):
        xout = []
        yout = []
        for sent in sents:
            xout.append([x for x, y in sent])
            yout.append([y for x, y in sent])
        yield np.array(xout), np.array(yout)

def train_evaluate_model(modelfn=None, epoch=None):
    """Trains and saves a NN model."""
    # Load data.
    data = mouth_data.MouthData(annotations_fn='mouthing.annotations2')

    # Load/create model.
    if modelfn:
        print('Loading NN model from {}'.format(modelfn))
        model = load_model(modelfn)
    else:
        print('Creating NN model')
        model = create_model()

    # Train model.
    print('Begin training')
    batch_size = 50
    initial_epoch = int(epoch) if epoch else 0
    callbacks = [
        ModelCheckpoint('mouthing-model-{epoch:02d}.hdf5', 'loss', save_best_only=True),
        EarlyStopping('loss', patience=5),
    ]
    model.fit_generator(get_data(data, batch_size),
        steps_per_epoch=int(len(data.sentences) / batch_size),
        epochs=initial_epoch + 100, callbacks=callbacks,
        initial_epoch=initial_epoch)

if __name__ == '__main__':
    # Usage: mouth_train.py [<model> [<initial epoch>]]
    modelfn, initial_epoch = (sys.argv + 2 * [None])[1:3]
    train_evaluate_model(modelfn, initial_epoch)
