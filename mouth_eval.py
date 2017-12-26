from argparse import ArgumentParser

from keras.models import load_model

from mouth_data import MouthData

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('model_file')
    args = argparser.parse_args()

    print('Loading model')
    model = load_model(args.model_file)

    data = MouthData()
    frames_data_bigrams = data.frames_resnet_bigrams()

    print('Classifying')
    prediction = model.predict(frames_data_bigrams, verbose=1)
    predicted_labels = (data.vector_to_label(vector) for vector in prediction)

    hypotheses_filename = 'hypotheses.txt'
    print('\nSaving results to {}'.format(hypotheses_filename))
    with open(hypotheses_filename, 'w') as hypotheses_file:
        for path, label in zip(data.paths, predicted_labels):
            hypotheses_file.write('{} {}\n'.format(path, label))
