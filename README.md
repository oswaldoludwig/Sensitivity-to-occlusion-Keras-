# Sensitivity to-occlusion using Keras
This is script evaluates the sensitivity of the VGG-16 pre-trained model to occlusion using Keras.
The idea is to check if the ConvNet is truly identifying the location of the object in the image by systematically occluding different portions of the image with a white square and evaluating the classifier output, see more details in Section 4.2 of: https://arxiv.org/pdf/1311.2901.pdf
This script reuses pieces of code belonging to this other script (which uses Caffe instead Keras):
https://github.com/BUPTLdy/occlusion_experiments/blob/master/Occlusion_experiments.ipynb
Set the path to the VGG-16 model weights in Line 22 of "occlusion_sensitivity.py". The VGG-16 file can be downloaded from: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
Set the the path to your image in Line 24 of "occlusion_sensitivity.py".
It's possible to evaluate the VGG-16 sensitivity to a specific object. To do so, you have to change the variable "index_object" in Line 130 of "occlusion_sensitivity.py" by the index of the class of interest. The VGG-16 output indices can be found here:
https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt
You can run this code in GPU by typing: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,exception_verbosity=high python occlusion_sensitivity.py
The result will be something like that:
![alt tag](https://github.com/oswaldoludwig/Sensitivity-to-occlusion-Keras-/blob/master/result.png)
