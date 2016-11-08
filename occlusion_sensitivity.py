'''This script was developed by Oswaldo Ludwig by reusing pieces of code from:
https://github.com/BUPTLdy/occlusion_experiments/blob/master/Occlusion_experiments.ipynb
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
import math
import pylab
import seaborn as sns
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from numpy.random import permutation
from keras.optimizers import SGD

# Enter here the path to the VGG16 model weights:
# This file can be downloaded from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
weights_path = '/home/oswaldo/video_classification/competitionKaggle/vgg16_weights.h5'
# Enter here the path to the image:
image_path = '/home/oswaldo/Documents/Zalando/blue_jean.jpg'
# Enter here the parameters of the occluding window:
occluding_size = 100
occluding_pixel = 0
occluding_stride = 5

print 'build the net...'

#The VGG16 model:

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000, activation='softmax'))

# You can set larger values here, according with the memory of your GPU:
batch_size = 32

print 'Loading the model weights...'
#loading the pre-trained VGG16 weights:
model.load_weights(weights_path)

print 'Compiling the model...'
# Compiling the model with a SGD/momentum optimizer:
model.compile(loss = "categorical_crossentropy",
              optimizer=optimizers.SGD(lr=1e-6, momentum=0.9),
              metrics=['mean_squared_logarithmic_error', 'accuracy'])

def Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride):
    
    image = cv2.imread(image_path)
    im = cv2.resize(image, (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    out = model.predict(im)
    out = out[0]
    # Getting the index of the winning class:
    m = max(out)
    index_object = [i for i, j in enumerate(out) if j == m]
    height, width, _ = image.shape
    output_height = int(math.ceil((height-occluding_size) / occluding_stride + 1))
    output_width = int(math.ceil((width-occluding_size) / occluding_stride + 1))
    heatmap = np.zeros((output_height, output_width))
    
    for h in xrange(output_height):
        for w in xrange(output_width):
            # Occluder region:
            h_start = h * occluding_stride
            w_start = w * occluding_stride
            h_end = min(height, h_start + occluding_size)
            w_end = min(width, w_start + occluding_size)
            # Getting the image copy, applying the occluding window and classifying it again:
            input_image = copy.copy(image)
            input_image[h_start:h_end, w_start:w_end,:] =  occluding_pixel            
            im = cv2.resize(input_image, (224, 224)).astype(np.float32)
            im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            out = model.predict(im)
            out = out[0]
            print('scanning position (%s, %s)'%(h,w))
            # It's possible to evaluate the VGG-16 sensitivity to a specific object.
            # To do so, you have to change the variable "index_object" by the index of
            # the class of interest. The VGG-16 output indices can be found here:
            # https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
            prob = (out[index_object]) 
            heatmap[h,w] = prob
    
    f = pylab.figure()
    f.add_subplot(1, 2, 0)  # this line outputs images side-by-side    
    ax = sns.heatmap(heatmap,xticklabels=False, yticklabels=False)
    f.add_subplot(1, 2, 1)
    plt.imshow(image)
    plt.show()
    print ( 'Object index is %s'%index_object)
    
Occlusion_exp(image_path, occluding_size, occluding_pixel, occluding_stride)