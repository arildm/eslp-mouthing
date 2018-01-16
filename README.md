ESLP Mouthing
=============

Train an RNN model on [Oscar Koller's](https://www-i6.informatik.rwth-aachen.de/~koller/) RWTH-PHOENIX Weather mouthing annotations.

Course project in Embodied and Situated Language Processing at University of Gothenburg 2017.

## Usage

Download and extract the mouthing archive (162 MB).

    curl ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-phoenix/2016/phoenix-mouthing-ECCV.tar.gz | tar xz

Extend dataset with variations of frames. This will create subdirectories of images and extra annotation files in the dataset folder.

    python3 facecropping-data-augmentation.py

Run frames through ResNet50 and save 7x7x2048 data chunks as npy files.

    python3 mouth_data.py

Train RNN model with Keras. Model files are created per epoch, the last one created is the best.

    python3 mouth_train.py

Run classification and write hypotheses file.

    python3 mouth_eval.py mouthing-model-XX.hdf5

Evaluate using Koller's eval script (Python 2).

    cd phoenix-mouthing-ECCV
    python2 evaluation/eval.py annotations/mouthing.annotations2gold ../hypotheses.txt

## Models

The repo contains three models M1, M2 and M3.
There are git tags for m1 and m2 while M3 lives in master.
The models have been trained and tested.
Trained models are too big for Git,
but hypothesis files have been written and renamed to m?.txt.

## Results

