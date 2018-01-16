from os import listdir, system
from fnmatch import fnmatch

for fn in listdir():
    if fnmatch(fn, 'm?.txt'):
        annfnext = {'m1.txt': '', 'm2.txt': '', 'm3.txt': '2gold'}[fn]
        for arg in ['', '--cleanCondition']:
            print(fn, arg)
            system('python2 phoenix-mouthing-ECCV/evaluation/eval.py {} phoenix-mouthing-ECCV/annotations/mouthing.annotations{} {}'
                .format(arg, annfnext, fn))
