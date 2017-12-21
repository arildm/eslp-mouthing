from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

MOUTHING_DATA_DIR = 'phoenix-mouthing-ECCV'

class ReversibleDict(dict):
    def __init__(self, *args):
        dict.__init__(self, *args)
        self.rev = dict()
        for k, v in self.items():
            if v in self.rev:
                raise ValueError('Repeated value not allowed: "{}"'.format(v))
            self.rev[v] = k

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.rev[value] = key

class LabelMap(ReversibleDict):
    def __init__(self, *args):
        ReversibleDict.__init__(self, *args)
        with open(MOUTHING_DATA_DIR + '/annotations/label-id') as map_file:
            for line in map_file.read().splitlines():
                id, label = line.split(' ')
                self[int(id)] = label

label_map = LabelMap()

def load_image_input(paths):
    img_arrays = [img_to_array(load_img(MOUTHING_DATA_DIR + path[2:], target_size=(224,224))) for path in paths]
    return preprocess_input(np.array(img_arrays))

def mouthing_data(limit=None, spread_labels=True):
    with open(MOUTHING_DATA_DIR + '/annotations/mouthing.annotations') as annotations_file:
        lines = annotations_file.read().splitlines()
        items = []
        for line in lines:
            path, labels_all = line.split(' ')
            labels = labels_all.split('-')
            items.append((path, labels[0]))
            # Break at limit.
            if limit is not None and len(items) >= limit:
                break
    return items

def resnet_convert(paths):
    print('Loading ResNet50')
    resnet = ResNet50(False)
    print('Loading images')
    images = load_image_input(paths)
    print('Processing images through ResNet50')
    return resnet.predict(images).reshape((len(images), 2048))

def mouthing_data_resnet():
    paths_data = mouthing_data()
    paths = (path for path, lalbel in paths_data)
    labels = [label_map.rev[label] for path, label in paths_data]

    resnet_data = resnet_convert(paths)
    # Add label item to each image resnet vector.
    cat_data = np.c_[resnet_data, labels]
    save_filename = 'mouthing-frames-resnet-labels.npy'
    print('Saving data of shape {} to {}'.format(cat_data.shape, save_filename))
    np.save(save_filename, cat_data)

mouthing_data_resnet()
