from argparse import ArgumentParser

from keras.models import load_model
import numpy as np
import sys

from mouth_data import MouthData
from mouth_train import get_data

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('model_file')
    args = argparser.parse_args()

    print('Loading model')
    model = load_model(args.model_file)

    # Load data.
    data = MouthData()
    x, y = get_data()

    print('Classifying')
    predicted_labels = []
    for i in range(len(x)):
        prediction = model.predict(np.array([x[i]]))
        sentlen = len(data.paths_by_sentence[data.sentences[i]])
        predicted_labels += [data.vector_to_label(vector) for vector in prediction[0][:sentlen]]
        sys.stdout.write('.')
        sys.stdout.flush()

    hypotheses_filename = 'hypotheses.txt'
    print('\nSaving results to {}'.format(hypotheses_filename))
    with open(hypotheses_filename, 'w') as hypotheses_file:
        for path, label in zip(data.paths, predicted_labels):
            hypotheses_file.write('{} {}\n'.format(path, label))
