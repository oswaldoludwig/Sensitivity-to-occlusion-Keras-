# Sensitivity to occlusion using Keras
This code evaluates the sensitivity of VGG-16 to occlusion using Keras. The idea is to check if the ConvNet is truly identifying the location of the object in the image by systematically occluding different portions of the image with a white square and evaluating the net output. For more details, see Section 4.2 of: https://arxiv.org/pdf/1311.2901.pdf

This code reuses pieces of code from: https://github.com/BUPTLdy/occlusion_experiments/blob/master/Occlusion_experiments.ipynb (which uses Caffe instead of Keras).

Set the path to the VGG-16 model weights in Line 22 of "occlusion_sensitivity.py". The VGG-16 file can be downloaded here: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

Set the the path to your image in Line 24 of "occlusion_sensitivity.py".
It's possible to evaluate the VGG-16 sensitivity to a specific class of objects. To do so, you have to change the variable "index_object" in Line 130 of "occlusion_sensitivity.py" by the index of the class of interest. The VGG-16 output indices can be found here:
https://github.com/HoldenCaulfieldRye/caffe/blob/master/data/ilsvrc12/synset_words.txt

The output of "occlusion_sensitivity.py" is something like this:
![alt tag](https://github.com/oswaldoludwig/Sensitivity-to-occlusion-Keras-/blob/master/result.png)
