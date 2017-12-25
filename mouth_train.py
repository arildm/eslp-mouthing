from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
import numpy as np

MOUTHING_DATA_DIR = 'phoenix-mouthing-ECCV'

def read_frames_data():
    """Loads the ResNet50-converted frame images."""
    data_filename = 'mouthing-frames-resnet.npy'
    resnet_data = np.load(data_filename)
    print('Loaded data of shape {} from {}'.format(resnet_data.shape, data_filename))
    return resnet_data

def load_label_map():
    """Loads labels as 1) a map to ids and 2) one-hot arrays of ids in order."""
    id_map = dict()
    with open(MOUTHING_DATA_DIR + '/annotations/label-id') as map_file:
        for line in map_file.read().splitlines():
            id, label = line.split(' ')
            id_map[label] = int(id)

    cats = to_categorical(sorted(range(max(id_map.values()) + 1)))
    return id_map, cats

def load_mouth_annotations():
    """Loads annotations, converts to onehot and returns as list."""
    label_map, label_onehots = load_label_map()
    with open(MOUTHING_DATA_DIR + '/annotations/mouthing.annotations') as annotations_file:
        lines = annotations_file.read().splitlines()
        items = []
        for line in lines:
            path, labels_all = line.split(' ')
            labels = labels_all.split('-')
            onehot = sum(label_onehots[label_map[label]] for label in labels) / len(labels)
            items.append(onehot)
        return items

def create_model(nlabels):
    """Creates an RNN model for sequential frame data input and label output."""
    model = Sequential()
    model.add(LSTM(64, input_shape=(2, 7, 7, 2048)))
    model.add(Dense(nlabels))
    model.add(Activation('softmax'))

    model.compile('adam', 'categorical_crossentropy', ['accuracy', 'recall'])
    return model
