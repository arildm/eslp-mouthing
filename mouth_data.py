from fnmatch import fnmatch
from itertools import chain
import os

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.engine.training import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
import numpy as np
from numpy.ma.core import argmax

class ResNet50Data:
    def convert(self, paths, data_filename='resnet-data-{}.npy'):
        print('Loading ResNet50')
        self.resnet = ResNet50(weights='imagenet', include_top=False)
        # Discard last pooling & activation layer.
        self.resnet = Model(self.resnet.inputs, self.resnet.layers[-2].output)

        paths = list(paths)
        print('Loading images from {0} paths'.format(len(paths)))

        for index in range(2**12, len(paths), 2**12):
            print('Loading images')
            img_arrays = [img_to_array(load_img(path, target_size=(224,224))) for path in paths[:index]]
            images = preprocess_input(np.array(img_arrays))

            print('Processing images through ResNet50')
            np.save(data_filename.format(index), self.resnet.predict(images))

        if len(paths[index:])>0:
            print('Loading images')
            img_arrays = [img_to_array(load_img(path, target_size=(224,224))) for path in paths[index:]]
            images = preprocess_input(np.array(img_arrays))

            print('Processing images through ResNet50')
            np.save(data_filename.format(len(paths)), self.resnet.predict(images))

        return

    def load(self, data_filename='resnet-data-{}.npy'):
        # chain is a concatenating generator: ['ABC', 'DEF'] -> 'ABCDEF'
        return chain.from_iterable(np.load(fn) for fn in os.listdir('.')
            if fnmatch(fn, data_filename.replace('{}', '*')))

class MouthData:
    """Contains frames input and labels in Keras-friendly formats."""
    def __init__(self, data_dir='phoenix-mouthing-ECCV', limit=None,
            annotations_fn ='mouthing.annotations'):
        self.data_dir = data_dir

        # Load annotations.
        self.annotations = []
        self.paths = []
        with open(self.data_dir + '/annotations/' + annotations_fn) as annotations_file:
            lines = annotations_file.read().splitlines()
            for line in lines:
                path, labels_all = line.split(' ')
                labels = labels_all.split('-')
                self.paths.append(path)
                self.annotations.append((path, labels))
                # Break at limit.
                if limit is not None and len(self.paths) >= int(limit):
                    break

        # Partition paths by sentence.
        self.sentences = []
        self.paths_by_sentence = dict()
        for path in self.paths:
            sentid = path.split('/')[-2]
            if sentid not in self.sentences:
                self.sentences.append(sentid)
                self.paths_by_sentence[sentid] = []
            self.paths_by_sentence[sentid].append(path)

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

        print('Loaded annotations for {} frames in {} sentences'
            .format(len(self), len(self.sentences)))

    def __len__(self):
        return len(self.paths)

    def frames_resnet(self):
        """Generates frame resnet data."""
        return ResNet50Data().load()

    def label_to_onehot(self, label):
        return self.label_onehots[self.id_map[label]]

    def vector_to_label(self, vector):
        # stochastic interpretation of probabilities
        # it takes a sample  with one run, but it can have more
        #vector = np.random.multinomial(1, vector).flatten()
        return list(self.id_map.keys())[list(self.id_map.values()).index(argmax(vector))]

    def annotation_vectors(self):
        """Loads annotations as vectors."""
        return [sum(self.label_to_onehot(label) for label in labels) / len(labels)
            for path, labels in self.annotations]

    def by_sentence(self, a, pad=156):
        print('Converting data of shape {} by sentence'.format(a.shape))
        shape = list(a.shape)
        sents = []
        i = 0
        for sentid in self.sentences:
            n = len(self.paths_by_sentence[sentid])
            sent = a[i:i+n]
            if pad:
                # Initially, shape[0] is data length, but since we're adding a
                # dimension, shape now has the dimensionality of each sentence,
                # and shape[0] is sentence length. Or here, the padding width.
                shape[0] = pad - n
                sent = np.concatenate([sent, (np.zeros(tuple(shape)))])
            sents.append(sent)
            i += n
        return np.array(sents)

if __name__ == '__main__':
    data = MouthData(annotations_fn='mouthing.annotations2')
    rel_paths = (data.data_dir + path[2:] for path in data.paths)
    ResNet50Data().convert(rel_paths)
