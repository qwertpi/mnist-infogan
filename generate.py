from math import ceil, sqrt
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from random import randint

from keras.engine.base_layer import Layer
from keras.initializers import RandomNormal, Constant
import keras.backend as K

class LeakyRelu(Layer):
    '''Like leaky relu but does leaky for above a threshold and below a theshold with learned thresholds and learned slopes'''

    def __init__(self, **kwargs):
        super(LeakyRelu, self).__init__(**kwargs)
        #aprox 90% chance of being 0.1-0.45 and aprox 50% chance of 0.2-0.35 and also aprox 50% chance of being 0.3-0.5 with overall the highest chance at 0.3
        self.alpha_initializer = RandomNormal(0.3, 0.1)
        self.__name__ = "LeakyRelu"
        if K.backend() == "tensorflow":
            from tensorflow import where
            #imports for the whole of the class not just this function and also renames to switch as I find it a more logical name
            self.switch = where
        #I use tensorflow and my very brief googling didn't reveal any equaivalent of tensorflow.where for theano
        elif K.backend() == "theano":
            print("Sorry only tensorflow is supported for LeakyRelu, if you know how to implement it in theano feel free to send a PR on github")
            
    #keras requires this takes the parameter input_shape even though we don't use it
    def build(self, input_shape):
        '''Called by Keras on init to set up the trainable paramters'''
        #makes alpha_a and alpha_b learnable PRelu style
        #seperate alphas for -1 leakage and 1 leakage
        #beta controls where the leakages start and is also trainable
        #each layer will have it's own sperate alpha_a, alpha_b, beta_a and beta_b as a seperate LeakyRelu object is used for each layer
        self.alpha_a = self.add_weight(name='alpha_a',
                                       shape=(1,),
                                       initializer=self.alpha_initializer,
                                       trainable=True)
        self.alpha_b = self.add_weight(name='alpha_b',
                                       shape=(1,),
                                       initializer=self.alpha_initializer,
                                       trainable=True)
        
        self.beta_a = self.add_weight(name='beta_a',
                                      shape=(1,),
                                      initializer=Constant(-1),
                                      trainable=True)
        self.beta_b = self.add_weight(name='beta_b',
                                      shape=(1,),
                                      initializer=Constant(1),
                                      trainable=True)
    def call(self, x):
        '''Where the main logic lives'''
        x = K.cast(x, "float32")
        #y=alpha*x+c rearnaged so at x=-1 the leaky component also outputs -1 (see https://www.desmos.com/calculator/fnsuod7zka)
        self.c_a = -1 * self.alpha_a * self.beta_a + self.beta_a
        #same as above but so at x=1 leaky outputs 1
        self.c_b = -1 * self.alpha_b * self.beta_b + self.beta_b
        '''
        This is the same as
        if x>-1:
            if x<1:
                return x
            else:
                return x*self.alpha_b+self.c_b
        else:
            x*self.alpha_a+self.c_a
        '''
        return self.switch(K.greater(x, self.beta_a), self.switch(K.less(x, self.beta_b), x, x*self.alpha_b+self.c_b), x*self.alpha_a+self.c_a)

    def compute_output_shape(self, input_shape):
        '''Called by keras so it knows what input shape the next layer can expect'''
        #keras requires an output shape be given and as an activation the output should be the same size as the input
        return input_shape
    
def reverse_tanh(x):
    '''
    Turns tanh activated data into 0-1
    :param x: the numpy array to be changed
    :return: the changed numpy array
    '''
    #makes -1=0 and 1=1
    #rounds to the nearest integer for pure black and white (no grays)
    return np.rint(x * 0.5 +0.5)

#loads the model from the saved model file
json_file = open('model.json', 'r')

mapping = {'LeakyRelu':LeakyRelu()}
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json, mapping)

# load weights into new model
model.load_weights("gen.h5")

#the number of images to generate
num = 144
while True:
    #generates the noise to be fed into the model
    noise = np.random.normal(0, 1, (num, 100))
    label = np.array([randint(0,10) for i in range(0, num)])
    #generates images
    gen_img = model.predict([noise, label])
    fig = plt.figure()
    #creates the matplotlib plot
    for i in range(0, num):
        fig.add_subplot(ceil(sqrt(num)), ceil(sqrt(num)), i+1)
        plt.imshow(reverse_tanh(gen_img[i].reshape(28,28)), cmap="gray", vmin=0, vmax=1, interpolation="lanczos")
        plt.title(label[i], {'fontsize':8})
        plt.axis("off")
    #fullscreen style
    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.96, wspace=0, hspace=0.3)
    #shows the plot
    plt.show()