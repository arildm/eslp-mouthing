import os

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.engine.training import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
from nltk import bigrams
import numpy as np
from numpy.ma.core import argmax

class ResNet50Data:
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def convert(self):
        print('Loading images')
        img_arrays = [img_to_array(load_img(path, target_size=(224,224))) for path in self.image_paths]
        images = preprocess_input(np.array(img_arrays))

        print('Loading ResNet50')
        resnet = ResNet50(weights='imagenet', include_top=False)
        # Discard last pooling & activation layer.
        resnet = Model(resnet.inputs, resnet.layers[-2].output)

        print('Processing images through ResNet50')
        return resnet.predict(images)

    def lazyload(self, data_filename='resnet-data.npy'):
        """Creates, or loads previously created, resnet-converted data."""
        if os.path.exists(data_filename):
            resnet_data = np.load(data_filename)
            print('Loaded data of shape {} from {}'.format(resnet_data.shape, data_filename))
        else:
            resnet_data = self.convert()
            print('Saving data of shape {} to {}'.format(resnet_data.shape, data_filename))
            np.save(data_filename, resnet_data)

        return resnet_data

class MouthData:
    """Contains frames input and labels in Keras-friendly formats."""
    def __init__(self, data_dir='phoenix-mouthing-ECCV', limit=None):
        self.data_dir = data_dir

        # Load annotations.
        self.annotations = []
        self.paths = []
        with open(self.data_dir + '/annotations/mouthing.annotations') as annotations_file:
            lines = annotations_file.read().splitlines()
            for line in lines:
                path, labels_all = line.split(' ')
                labels = labels_all.split('-')
                self.paths.append(path)
                self.annotations.append((path, labels))
                # Break at limit.
                if limit is not None and len(self.paths) >= int(limit):
                    break

        # Load label/id mapping.
        self.id_map = dict()
        with open(self.data_dir + '/annotations/label-id') as map_file:
            for line in map_file.read().splitlines():
                id, label = line.split(' ')
                # Their ids in the file are 1-indexed. Switch to 0-indexed
                # by subtracting 1, so our id = index.
                self.id_map[label] = int(id) - 1

        # Create list of one-hot vectors (as a list, it "maps" ids to vectors).
        self.label_onehots = to_categorical(sorted(self.id_map.values()))

    def __len__(self):
        return len(self.paths)

    def frames_resnet(self):
        """Get ResNet50 representations of frame images."""
        rel_paths = (self.data_dir + path[2:] for path in self.paths)
        resnet_data = ResNet50Data(rel_paths)
        return resnet_data.lazyload('mouthing-frames-resnet.npy')

    def frames_resnet_bigrams(self):
        frames_data = self.frames_resnet()
        # Duplicate first element to provide dummy bigram for the first sample.
        frames_data = np.insert(frames_data, 0, frames_data[0], axis=0)
        return np.array(list(bigrams(frames_data)))

    def label_to_onehot(self, label):
        return self.label_onehots[self.id_map[label]]

    def vector_to_label(self, vector):
        return list(self.id_map.keys())[list(self.id_map.values()).index(argmax(vector))]

    def annotation_vectors(self):
        """Loads annotations as vectors."""
        return [sum(self.label_to_onehot(label) for label in labels) / len(labels)
            for path, labels in self.annotations]

if __name__ == '__main__':
    data = MouthData()
    data.frames_resnet()
