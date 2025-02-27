# TensorFlow2-Keras-Michelucci-2022
Examples based on the Umberto Michelucci's book: "Applied Deep Learning with TensorFlow 2: Learn to Implement Advanced Deep Learning Techniques with Python", Second Edition, Apress, 2022

## Online sources

Book's online version:
https://adl.toelt.ai

Original GitHub by TOELT LLC:
https://github.com/toelt-llc/BOOK-ADL-Book-2nd-Ed

## Datasets

Radon activity: http://www.stat.columbia.edu/~gelman/arm/examples/radon/

BCCD (blood cells): https://github.com/ax-va/BCCD_Dataset/tree/master/BCCD/Annotations/

Zalando clothes images: https://www.kaggle.com/zalando-research/fashionmnist/data; https://github.com/zalandoresearch/fashion-mnist (convert with  https://pjreddie.com/projects/mnist-in-csv/); or download from keras.datasets (used in code):
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
```

MNIST handwritten digits: http://yann.lecun.com/exdb/mnist/; or download from Keras: 
```python
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
```

Boston area housing: http://lib.stat.cmu.edu/datasets/boston; https://archive.ics.uci.edu/ml/machine-learning-databases/housing/
 
http://www.image-net.org/

## Useful links

IPython documentation:
https://ipython.readthedocs.io/en/st

Kaggle competitions:
https://www.kaggle.com/competitions
