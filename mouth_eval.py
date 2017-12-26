from argparse import ArgumentParser

from keras.models import load_model
import numpy as np
import sys

from mouth_data import MouthData

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('model_file')
    args = argparser.parse_args()

    print('Loading model')
    model = load_model(args.model_file)

    data = MouthData()
    frames_data_bigrams = data.frames_resnet_bigrams()

    hypotheses_filename = 'hypotheses.txt'
    print('Classifying, saving results to {}'.format(hypotheses_filename))
    with open(hypotheses_filename, 'w') as hypotheses_file:
        for i in range(len(data)):
            prediction = model.predict(np.array([frames_data_bigrams[i]]))
            label = data.vector_to_label(prediction)
            path = data.paths[i]
            sys.stdout.write(label + ' ')
            hypotheses_file.write('{} {}\n'.format(path, label))
    print()
