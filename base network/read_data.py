'''
This code from Michael Nielsen's book "Neural Networks and Deep Learning".
A few modifications have been made to make it compatible with Python 3.6.
Additionally, transformations have been added to the dataset to train the network with more accuracy.
'''

import _pickle as cPickle
import gzip
import scipy.ndimage as ndimage
import numpy as np

def load_data():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = cPickle.load(f, encoding='latin1')
    f.close()
    return (training_data, validation_data, test_data)

def rotate_image(image, angle):
    return ndimage.rotate(image.reshape(28, 28), angle, reshape=False).flatten()

def scale_image(image, scale):
    image = image.reshape(28, 28)
    
    zoom_factor = (scale, scale)
    scaled_image = ndimage.zoom(image, zoom_factor, order=1)

    if scaled_image.shape[0] != 28 or scaled_image.shape[1] != 28:
        scaled_image = ndimage.zoom(scaled_image, (28/scaled_image.shape[0], 28/scaled_image.shape[1]), order=1)
    
    return scaled_image.flatten()

def shift_image(image, shift):
    return ndimage.shift(image.reshape(28, 28), shift).flatten()

def transform_image(image):
    image = rotate_image(image, np.random.uniform(-30, 30))
    image = scale_image(image, np.random.uniform(0.9, 1.1))
    image = shift_image(image, [np.random.uniform(-4, 4), np.random.uniform(-2, 2)])

    return image

def load_data_wrapper(augment_data=False):
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]

    if augment_data:
        augmented_inputs = []
        augmented_results = []
        for img, result in zip(training_inputs, training_results):
            transformed_img = transform_image(img)
            augmented_inputs.append(transformed_img.reshape(784, 1))
            augmented_results.append(result)

        training_inputs.extend(augmented_inputs)
        training_results.extend(augmented_results)

    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e