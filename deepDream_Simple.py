from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input
from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from PIL import Image

'''
Simple Deep dream application based on VGG16 classifier, pretrained on imagenet dataset
Bare bones implementation which can show some interesting results even if only running on CPU
Written by Grant Holtes, December 2018
www.grantholtes.com
'''

def get_feature_reps(layer_name, model):
    selectedLayer = model.get_layer(layer_name)
    return selectedLayer.output

def loss(layer_name, model):
    return - K.sum(get_feature_reps(layer_name, model))

def grad(x, layer_name, model):
    gradFunc = K.function([model.input], K.gradients(loss(layer_name, model), [model.input]))
    return gradFunc([x])

def postprocess_array(x):
    x[..., 0] += 103.939
    x[..., 1] += 116.779
    x[..., 2] += 123.68
    # 'BGR'->'RGB'
    x = x[..., ::-1]
    x = np.clip(x, 0, 255)
    x = x.astype('uint8')
    return x

n = 10 #Define Learbing rate
iterations = 20
maxDim = 200 #Max side length - larger increases computation time
layer_name = "block5_pool" #Which layer to maximise

tf_session = K.get_session()
Model = VGG16(include_top=False, weights='imagenet')
print("Model Loaded")

original = Image.open('data.jpg')
originalSize = original.size
#Determine if landscape or not
if originalSize[0] > originalSize[1]:
    w = maxDim
    h = round(maxDim*originalSize[1] / originalSize[0])
else:
    w = round(maxDim*originalSize[0] / originalSize[1])
    h = maxDim

resized = original.resize((w, h)) #shrink image to size as in maxDim
resizedArr = np.array(resized)    #Convert to numpy array
resizedArr = preprocess_input(np.expand_dims(resizedArr, axis=0)) #Preprocess for vgg16

#Perform gradient accent
for iter in range(iterations):
    print("Iteration: {0}".format(iter+1))
    gradIter = grad(resizedArr, layer_name, Model)
    resizedArr = np.add(resizedArr, -np.multiply(gradIter, n))[0]

#Convert back to a RGB image and save as original size
dream = Image.fromarray(postprocess_array(resizedArr[0]))
dream = dream.resize(originalSize)
dream.save("dream.jpg")
