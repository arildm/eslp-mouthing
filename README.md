ESLP Mouthing
=============

Train an RNN model on [Oscar Koller's](https://www-i6.informatik.rwth-aachen.de/~koller/) RWTH-PHOENIX Weather mouthing annotations.

Course project in Embodied and Situated Language Processing at University of Gothenburg 2017.

## Usage

Download and extract the mouthing archive (162 MB).

    curl ftp://wasserstoff.informatik.rwth-aachen.de/pub/rwth-phoenix/2016/phoenix-mouthing-ECCV.tar.gz | tar xz

Run frames through ResNet50 and save 7x7x2048 data as a npy file.

    python3 mouth_data.py

Train RNN model with Keras. Model files are created per epoch, the last one created is the best.

    python3 mouth_train.py

Run classification and write hypotheses file.

    python3 mouth_eval.py mouthing-model-XX.hdf5

Evaluate using Koller's eval script (Python 2).

    cd phoenix-mouthing-ECCV
    python2 evaluation/eval.py annotations/mouthing.annotations ../hypotheses.txt

## Results

    Precision: 38.6554621849 Recall: 2.48514316586 Evaluated 3687 frames

