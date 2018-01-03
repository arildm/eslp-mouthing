import os

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.engine.training import Model
from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array
from keras.utils.np_utils import to_categorical
from nltk import bigrams
import numpy as np
from numpy.ma.core import argmax

class ResNet50Data:
    def convert(self, paths):
        print('Loading images')
        img_arrays = [img_to_array(load_img(path, target_size=(224,224))) for path in paths]
        images = preprocess_input(np.array(img_arrays))

        print('Loading ResNet50')
        self.resnet = ResNet50(weights='imagenet', include_top=False)
        # Discard last pooling & activation layer.
        self.resnet = Model(self.resnet.inputs, self.resnet.layers[-2].output)

        print('Processing images through ResNet50')
        return self.resnet.predict(images)

    def lazyload(self, paths, data_filename='resnet-data.npy'):
        """Creates, or loads previously created, resnet-converted data."""
        if os.path.exists(data_filename):
            resnet_data = np.load(data_filename)
            print('Loaded data of shape {} from {}'.format(resnet_data.shape, data_filename))
        else:
            resnet_data = self.convert(paths)
            print('Saving data of shape {} to {}'.format(resnet_data.shape, data_filename))
            np.save(data_filename, resnet_data)

        return resnet_data

    def image_data_generator(self, x, y, *args):
        gen = ImageDataGenerator(*args)
        for item in gen.flow(x, y):
            return item

class MouthData:
    """Contains frames input and labels in Keras-friendly formats."""
    def __init__(self, data_dir='phoenix-mouthing-ECCV', limit=None, pad=None):
        self.data_dir = data_dir
        self.pad = pad

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

        # Partition paths by sentence.
        self.sentences = []
        self.paths_by_sentence = dict()
        for path in self.paths:
            sentid = path.split('/')[-2]
            if sentid not in self.sentences:
                self.sentences.append(sentid)
                self.paths_by_sentence[sentid] = []
            self.paths_by_sentence[sentid].append(path)

        # By default, pad to longest sentence.
        if self.pad is None:
            self.pad = max(len(sent) for sent in self.paths_by_sentence.values())

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

    def load_images(self, paths):
        img_arrays = [img_to_array(load_img(path, target_size=(224,224))) for path in paths]
        return preprocess_input(np.array(img_arrays))

    def frames(self):
        """Keras-ready image data for all frames."""
        rel_paths = (self.data_dir + path[2:] for path in self.paths)
        return self.load_images(rel_paths)

    def frames_by_sentence(self):
        """Keras-ready image data for frames by sentence."""
        # More memory-friendly than self.by_sentence(self.frames()).
        for sentid, paths in self.paths_by_sentence.items():
            rel_paths = (self.data_dir + path[2:] for path in paths)
            sent = self.load_images(rel_paths)
            # Shorten if pad < len(sent), else pad it. No `if` actually needed.
            sent = sent[:self.pad]
            padshape = tuple([self.pad - len(sent)] + list(sent.shape)[1:])
            sent = np.concatenate([sent, (np.zeros(tuple(padshape)))])
            yield np.array(sent)

    def frames_resnet(self):
        """Get ResNet50 representations of frame images."""
        rel_paths = (self.data_dir + path[2:] for path in self.paths)
        resnet_data = ResNet50Data()
        return resnet_data.lazyload(rel_paths)

    def frames_resnet_bigrams(self):
        """Get ResNet50 representations of frames as bigrams."""
        frames_data = self.frames_resnet()
        # Duplicate first element to provide dummy bigram for the first sample.
        frames_data = np.insert(frames_data, 0, frames_data[0], axis=0)
        return np.array(list(bigrams(frames_data)))

    def label_to_onehot(self, label):
        return self.label_onehots[self.id_map[label]]

    def labels_to_vector(self, labels):
        return sum(self.label_to_onehot(label) for label in labels) / len(labels)

    def vector_to_label(self, vector):
        # stochastic interpretation of probabilities
        # it takes a sample  with one run, but it can have more
        #vector = np.random.multinomial(1, vector).flatten()
        return list(self.id_map.keys())[list(self.id_map.values()).index(argmax(vector))]

    def annotation_vectors(self):
        """Loads annotations as vectors."""
        return [self.labels_to_vector(labels)
            for path, labels in self.annotations]

    def by_sentence(self, a):
        print('Converting data of shape {} by sentence'.format(a.shape))
        sents = []
        i = 0
        for sentid in self.sentences:
            n = len(self.paths_by_sentence[sentid])
            sent = a[i:i+n]
            # Shorten or pad sent. No `if` actually needed.
            sent = sent[:self.pad]
            padshape = tuple([self.pad - len(sent)] + list(sent.shape)[1:])
            sent = np.concatenate([sent, (np.zeros(padshape))])
            sents.append(sent)
            i += n
        return np.array(sents)

    def data_generator(self, batch_size):
        """Generates data indefinitely, with random variation of images."""
        # todo Check IDG params.
        image_data_generator = ImageDataGenerator(rotation_range=2, zoom_range=.01, horizontal_flip=True)
        vectors = self.by_sentence(np.array(self.annotation_vectors()))
        while True:
            images = self.frames_by_sentence()
            for x, y in zip(images, vectors):
                # x is (pad) images, y is (pad) labels
                idg_batch = image_data_generator.flow(x, y, batch_size=batch_size)
                # In ImageDataGenerator each item is an image. In our own model,
                # each item is a sentence (156 images).
                batch_x = []
                batch_y = []
                for i, (idg_x, idg_y) in enumerate(idg_batch):
                    if i >= batch_size:
                        break
                    batch_x.append(idg_x)
                    batch_y.append(idg_y)
                yield (np.array(batch_x), np.array(batch_y))
