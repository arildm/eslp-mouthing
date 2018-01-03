from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.training import Model
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential

import mouth_data

def create_model(pad=156):
    """Creates an RNN model for sequential frame data input and label output."""
    model = Sequential()
    # Start off with ResNet50. Remove last layer, to get 7x7x2048 output.
    resnet = ResNet50(weights='imagenet', include_top=False)
    model.add(TimeDistributed(Model(resnet.inputs, resnet.layers[-2].output), input_shape=(pad, 224, 224, 3)))
    # Flatten input for LSTM.
    model.add(TimeDistributed(Flatten()))
    # LSTM helps with sequential data.
    model.add(LSTM(32, return_sequences=True))
    # Output layer the size of the number of labels.
    model.add(TimeDistributed(Dense(40)))
    # Softmax and categorical cross entropy are good for classification.
    model.add(Activation('softmax'))
    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.summary()
    return model

'''
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
'''

def train_evaluate_model():
    """Trains and saves a NN model."""
    # Load data.
    print('Loading data')
    data = mouth_data.MouthData()

    # Train model.
    print('Creating NN model')
    # Need to call set_learning_phase() to fix "You must feed a value for
    # placeholder tensor" error (depends on Tensorflow version?)
    K.set_learning_phase(0)
    model = create_model(data.pad)
    K.set_learning_phase(1)
    print('Begin training')
    batch_size = 5
    model.fit_generator(data.data_generator(batch_size), len(data) / batch_size,
        epochs=100, callbacks=[
        ModelCheckpoint('mouthing-model-{epoch:02d}.hdf5', 'loss',
            save_best_only=True),
        EarlyStopping('loss', patience=5),
    ])

if __name__ == '__main__':
    train_evaluate_model()
