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

The project contains three models: M1, M2 and M3.
There are git tags for the code for m1 and m2.
M3 lives in master.
The models have been trained and tested.
Trained models are too big for Git,
but model test output files (hypothesis files)
have been written to m?.txt.
A helper script phoenix-eval.py can be used to evaluate these,
automating the final step above.

## Results

          | Standard              | No garbage
    Model | Precision    | Recall | Precision  | Recall
    ---   | ---          | ---    | ---        | ---
    M1    | 19.4         | 12.7   | 20.6       | 12.2
    M2    | 68.7         | 62.8   | 67.8       | 60.2
    M3    | 43.5         | 10.4   | 43.0       | 9.9
