from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.engine.training import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.np_utils import to_categorical
import numpy as np

MOUTHING_DATA_DIR = 'phoenix-mouthing-ECCV'

def load_image_input(paths):
    img_arrays = [img_to_array(load_img(MOUTHING_DATA_DIR + path[2:], target_size=(224,224))) for path in paths]
    return preprocess_input(np.array(img_arrays))

def mouthing_data(limit=None):
    with open(MOUTHING_DATA_DIR + '/annotations/mouthing.annotations') as annotations_file:
        lines = annotations_file.read().splitlines()
        items = []
        for line in lines:
            path, labels_all = line.split(' ')
            labels = labels_all.split('-')
            items.append((path, labels[0]))
            # Break at limit.
            if limit is not None and len(items) >= int(limit):
                break
    return items

def resnet_convert(paths):
    print('Loading ResNet50')
    resnet = ResNet50(weights='imagenet', include_top=False)
    # Discard last pooling & activation layer.
    resnet = Model(resnet.inputs, resnet.layers[-2].output)

    print('Loading images')
    images = load_image_input(paths)
    print('Processing images through ResNet50')
    return resnet.predict(images)

def mouthing_data_resnet(limit=None):
    paths_data = mouthing_data(limit)
    paths = (path for path, lalbel in paths_data)

    resnet_data = resnet_convert(paths)
    save_filename = 'mouthing-frames-resnet.npy'
    print('Saving data of shape {} to {}'.format(resnet_data.shape, save_filename))
    np.save(save_filename, resnet_data)

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
            # Their ids in the file are 1-indexed. Switch to 0-indexed by
            # subtracting 1, so our id = index.
            id_map[label] = int(id) - 1

    cats = to_categorical(sorted(id_map.values()))
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
