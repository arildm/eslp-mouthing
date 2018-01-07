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

    # Load gold data inputs.
    data = MouthData(annotations_fn='mouthing.annotations2gold')
    inputs_gen = (a for a, b in get_data(data, 5))

    print('Classifying')
    predicted_labels = []
    prediction = model.predict_generator(inputs_gen,
        steps=len(data.sentences), verbose=1)
    for i, sentid in enumerate(data.sentences):
        sentlen = len(data.paths_by_sentence[sentid])
        predicted_labels += [data.vector_to_label(vector)
            for vector in prediction[i][:sentlen]]

    hypotheses_filename = 'hypotheses.txt'
    print('\nSaving results to {}'.format(hypotheses_filename))
    with open(hypotheses_filename, 'w') as hypotheses_file:
        for path, label in zip(data.paths, predicted_labels):
            hypotheses_file.write('{} {}\n'.format(path, label))
