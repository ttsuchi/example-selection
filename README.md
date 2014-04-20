ExampleSelection (Python)
============
Active Example Selection Project

## How to checkout

Initially I did

```
git remote add guru git@guru.ucsd.edu:ExampleSelection.git
git push -u guru master
```

To clone the project, do

```
git clone git@guru.ucsd.edu:ExampleSelection.git
cd ExampleSelection
```

## Dependencies

This software also depends on

Munkres [https://pypi.python.org/pypi/munkres/](https://pypi.python.org/pypi/munkres/)

pandas [http://pandas.pydata.org/](http://pandas.pydata.org/)

spams [http://spams-devel.gforge.inria.fr/](http://spams-devel.gforge.inria.fr/)

my fork of SaliencyMap [https://github.com/ttsuchi/saliency-map](https://github.com/ttsuchi/saliency-map)


### Installation

SaliencyMap depends on OpenCV. On MacPorts, this can be installed by

```
sudo port install atlas
sudo port install opencv +python27
```


Install pandas, munkres and SaliencyMap with:

```
sudo pip install pandas munkres
sudo pip install git+https://github.com/ttsuchi/saliency-map
```

SPAMS needs to be manually downloaded and compiled.  To compile SPAMS on OSX 10.9:

```
sudo ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future python setup.py install
```

on python27-apple and apple-gcc42.

Note: to install SPAMS on MacPorts-installed Python (that's under ```/opt/local```, have to change ```setup.py``` of SPAMS package to read:

```
libs = ['stdc++', 'blas', 'lapack', 'tatlas' ] # Added 'tatlas'

if osname.startswith("macosx"):
    cc_flags = ['-fPIC', '-fopenmp']
    link_flags = ['-F/opt/local/Library/Frameworks/', '-framework', 'Python']
```

first.)

Let's check the installation with

```
python -c "import sys; print sys.path; import numpy; print numpy.__file__; print numpy.__version__; import cv2; import spams"
```

## How to run the experiments

To run experiments, just execute run.py in the src/ directory:

```
./run.py [name] [subname] [number of iterations]
```

For instance, run the short demo:

```
./run.py demo short 1000
```

The "name" corresponds to the module name under the "experiment." package, which defines what kind of experiment to run. So this runs the experiment defined by "experiment.demo" module, and saves the result in results/demo-short.pkl file.

To execute in parallel mode, first start the IPython cluster:

```
ipcluster start -n 5 --daemon
```

Then run the code using the parallel executor:

```
./run.py -p small parallel 1000
```


For debugging library issues:

```
python -c "import sys; print sys.path; import numpy; print numpy.__file__; print numpy.__version__; import cv2; import spams"
```