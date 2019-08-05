from os import system as bash
from math import ceil, floor, sqrt
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, BatchNormalization, Reshape, Flatten, Dropout, Concatenate
from keras.utils.generic_utils import get_custom_objects
from keras.utils import to_categorical
from keras.optimizers import Adam
import keras.backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
#change this to from tqdm import tqdm_notebook as tqdm if using a juptyer notebook
from tqdm import tqdm
from keras.utils import plot_model

#imports for LeakyRelu custom relu
from keras.engine.base_layer import Layer
from keras.initializers import RandomNormal, Constant

K.clear_session()
#if train.py is outside of the folder images, model files etc. should be saved to, specify the relative path from train.py to the folder here (ensure to include a trailing slash)
path = ""

class LeakyRelu(Layer):
    '''Like leaky relu but does leaky for above a threshold and below a theshold with learned thresholds and learned slopes'''

    def __init__(self, **kwargs):
        super(LeakyRelu, self).__init__(**kwargs)
        
        self.alpha_initializer = RandomNormal(0.25, 0.1)
        self.__name__ = "LeakyRelu"
            
    #keras requires this takes the parameter input_shape even though we don't use it
    def build(self, input_shape):
        '''Called by Keras on init to set up the trainable paramters'''
        #makes alpha_a and alpha_b learnable PRelu style
        #seperate alphas for -1 leakage and 1 leakage
        #each layer will have it's own sperate alpha_a and alpha_b
        #starts like a fairly normal leakyrelu
        self.alpha_a = self.add_weight(name='alpha_a',
                                       shape=(1,),
                                       initializer=self.alpha_initializer,
                                       trainable=True)
        self.alpha_b = self.add_weight(name='alpha_b',
                                       shape=(1,),
                                       initializer=Constant(1),
                                       trainable=True)

    def call(self, x):
        '''Where the main logic lives'''
        x = K.cast(x, "float32")
        '''
        This is more or less equivalnet to same as
        if x>-1:
            if x<1:
                return x
            else:
                return x * self.alpha_b 
        else:
            x * self.alpha_a
        '''
        #see https://www.desmos.com/calculator/cpedmjbox1 for an interactive demo of this equation
        def neg_leaky(x):
            return K.relu(x + 1) - 1
        def neg_and_pos_leaky(x):
            return -1 * K.relu(-1 * neg_leaky(x) + 1) + 1
        return neg_and_pos_leaky(x) + (neg_leaky(-1 * x - 2) + 1) * - 1 * self.alpha_a + neg_and_pos_leaky(x) + (neg_leaky(x - 2) + 1) * - 1 * self.alpha_b

    def compute_output_shape(self, input_shape):
        '''Called by keras so it knows what input shape the next layer can expect'''
        #keras requires an output shape be given and as an activation the output should be the same size as the input
        return input_shape

get_custom_objects().update({'LeakyRelu': LeakyRelu()})

discrim_optimizer = Adam(0.0005, beta_1=0.5)
gen_optimizer = Adam(0.0005, beta_1=0.5)
classifier_optimizer = Adam(0.0005, beta_1=0.5)

#loads the MNIST data which we will be training the GAN to imitate
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#data normalisation to -1 to 1
x_train = x_train.astype('float32')/127.5 - 1
x_test = x_test.astype('float32')/127.5 - 1

#adding channels to shape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

#merging into one big X and one big Y as we don't care about validation data for unsupervised learning
X = np.concatenate((x_train, x_test))
Y = np.concatenate((y_train, y_test))

#generator portion of network
#takes an input of a 1D list of 100 numbers
noise_in = Input(shape=(100, ))
gen_label_in = Input(shape=(1, ))

#downcasts the 101D concatanted noise and label vectors into 100D space
dense = Dense(100)(Concatenate()([noise_in, gen_label_in]))
activation = LeakyRelu()(dense)

#12544 = 7*7*256 allowing the reshape to take place
dense = Dense(12544)(activation)
activation = LeakyRelu()(dense)
dropout = Dropout(0.5)(activation)
#reshapes to form a 7x7 image with 256 channels ready for convolutional layers
reshape = Reshape((7, 7, 256))(dropout)

#by making the kernel_size a multiple of the strides checkerbaord patterns are minimised (https://distill.pub/2016/deconv-checkerboard/ and https://arxiv.org/pdf/1609.05158.pdf)
#stride 2 for transpose conv results in an output size twice as big ie 14x14
conv = Conv2DTranspose(256, strides=2, kernel_size=4, padding="same")(reshape)
activation = LeakyRelu()(conv)
#batch norm is used on all layers apart from the first and last ones
norm = BatchNormalization()(activation)

#normal conv to break up transpose convs further minimising checkerboard patterns
conv = Conv2D(128, kernel_size=4, padding="same", activation="LeakyRelu")(norm)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

#transpose conv to upsample to 28x28 ie MNIST resolution
conv = Conv2DTranspose(64, kernel_size=4, strides=2, padding="same")(norm)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

conv = Conv2DTranspose(32, kernel_size=4, padding="same")(norm)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

#1 as we only want 1 channel ie grayscale output
#tanh to restrict output to -1 to 1
img_out = Conv2D(1, kernel_size=5, padding="same", activation="tanh")(norm)

generator = Model(inputs=[noise_in, gen_label_in], outputs=img_out)
#saves a diagram of the model
plot_model(generator, to_file=path+'generator.png', show_shapes=True)

#discriminator portion of network
#takes a 28x28x1 input ie a 28x28 grayscale image
img_in = Input((28, 28, 1))
disc_label_in = Input(shape=(1,))

#mirror of generator in terms of number of filters
conv = Conv2D(32, kernel_size=3, padding="same")(img_in)
activation = LeakyRelu()(conv)

#strides 2 to perform downsampling
conv = Conv2D(64, kernel_size=4, padding="same", strides=2)(activation)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

conv = Conv2D(128, kernel_size=4, padding="same", strides=2)(norm)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

conv = Conv2D(256, kernel_size=4, padding="same")(norm)
activation = LeakyRelu()(conv)
norm = BatchNormalization()(activation)

flatten = Flatten()(norm)

#the model has two outputs one outputs a 0 or 1 for real or fake
discrim_out = Dense(1)(flatten)
activation = LeakyRelu()(discrim_out)

#the other outputs a one hot encoded class prediction
class_out = Dense(10, activation='softmax')(flatten)

#creates discirmnator and classifier models
discriminator = Model(inputs=img_in, outputs=discrim_out)
classifier = Model(inputs=img_in, outputs=class_out)
discriminator.compile(loss='mse', optimizer=discrim_optimizer)
classifier.compile(loss='categorical_crossentropy', optimizer=classifier_optimizer)
plot_model(discriminator, to_file=path+'discriminator.png', show_shapes=True)
plot_model(classifier, to_file=path+'classifier.png', show_shapes=True)

#full model
#joins the discrimnator and the classifier with the output of the generator
#freezes discrimantor for the purposes of adding it to the combined model as the combined model is what will be used to train the generator
discriminator.trainable = False
classifier.trainable = False
#feeds the generator output layer into the discriminator and gets the ouput of that
discrim_out = discriminator(img_out)
#feeds the classifier output layer into the discriminator and gets the ouput of that
label_out = classifier(img_out)
#goes from the generator inputs to the discrimantor output and the label output ie every layer
combined = Model(inputs=[noise_in, gen_label_in], outputs=[discrim_out, label_out])
combined.compile(loss=['mse', 'categorical_crossentropy'], optimizer=gen_optimizer)

plot_model(combined, to_file=path+'model.png', show_shapes=True)

#saves the generator archtierure as a json file
#done here as the architecture doesn't change
model_json = generator.to_json()
with open(path+'model.json', "w") as json_file:
    json_file.write(model_json)
json_file.close()

#change to if False when doing the first train run after an archticuture change or if you for any other reason don't want to load the checkpoint weights
if True:
    #if the generator weights file exists load it
    try:
        with open(path+"gen.h5", 'r') as f:
            generator.load_weights(path+"gen.h5")
            print("Loaded generator checkpoint")
    except:
        pass
#change to if False when doing the first train run after an archticuture change or if you for any other reason don't want to load the checkpoint weights
if True:
    #if the discriminator weights file exists load it
    try:
        with open(path+"disc.h5", 'r') as f:
            discriminator.load_weights(path+"disc.h5")
            print("Loaded discriminator checkpoint")
    except:
        pass
    
    #if the classifier weights file exists load it
    try:
        with open(path+"class.h5", 'r') as f:
            classifier.load_weights(path+"class.h5")
            print("Loaded classifier checkpoint")
    except:
        pass

if True:
    try:
        with open(path+"epoch.txt", 'r') as f:
            epoch = int(f.read())
    except:
        epoch = 0
else:
    epoch = 0

#this is 98 as the models are trained on 49 real and 49 fake
batch_size = 49
while True:
    #generating y data
    #fake is 1 and valid is 0 to help with training
    #random numbers in range 0.9-1 and 0-0.1 respecitvely to help with training by prevening the discriminator from getting too confident
    fake = np.random.uniform(0.9, 1, (batch_size, 1))
    valid = np.random.uniform(0, 0.1, (batch_size, 1))
    
    #picks random numbers to decide which images to use
    index = np.random.randint(0, X.shape[0], batch_size)
    #gets the images with the indexes of those numbers
    images = X[index]

    #generates the noramlly dsitirbuted noise to be inputed
    noise = np.random.normal(0, 1, (batch_size, 100))
    #picks random labels to be genrated
    labels = np.random.randint(0, 10, (batch_size))
    
    #gets the generator to generate images for the noise and labels
    gen_images = generator.predict([noise, labels])
    
    
    #trains the discrimanator on real and fake data
    #sepearte batches to help batch norm (I think)
    disc_loss_1 = discriminator.train_on_batch(images, valid)
    disc_loss_2 = discriminator.train_on_batch(gen_images, fake)
    
    #trains the classifier with the one hot enecoded classes of the real images
    class_loss = classifier.train_on_batch(images, to_categorical(Y[index], 10))
    
    #trains the combined model (which only has generator layers trainable so we are only really training the generator) 
    #to make the discrimantor output 0 (valid) when passed generator images 
    #and make the classifier predict the correct class 
    #ie. make convicning images of the specified class
    gen_loss = combined.train_on_batch([noise, labels], [np.zeros((batch_size, 1)), to_categorical(labels, 10)])
    
    #if the epoch is a multiple of 100
    if epoch%100 == 0:
        #tells us the losses
        #gen_loss gives the loss for the discrimnaor output, clasifiee routpu and total in the order [total, discrinanor, classifier]
        #new line at the end to make it easier to tell where one epoch ends and another begins
        print(disc_loss_1, disc_loss_2, class_loss, gen_loss, "\n")
        
        with open("epoch.txt", "w") as f:
            #checkpoint the epoch so the epoch  number is correct when restarting
            f.write(str(epoch))
        
        #checkpoint the weights
        generator.save_weights(path+"gen.h5")
        discriminator.save_weights(path+"disc.h5")
        classifier.save_weights(path+"class.h5")
        
        #recreates the progress bar else it gets stuck at the top of the output
        progress_bar = tqdm(unit='', initial=epoch)
        
        #if the epoch is a multiple of 500
        #saves the generator images
        if epoch%500 == 0:
            #reversing tanh to make it 0-1 again
            gen_imgs = gen_images*0.5 + 0.5
            
            #spagtehti code coming up
            count = 0
            #creates a subplot of the square root of the length of gen_images rounded up by square root of the length of gen_images rounded up
            figure, axs = plt.subplots(ceil(sqrt(len(gen_imgs))), ceil(sqrt(len(gen_imgs))))
            #creates the square root of the length of gen_images rounded up rows
            for row in range(0, ceil(sqrt(len(gen_imgs)))):
                #square root of the length of gen_images rounded down images
                for image in range(0, floor(sqrt(len(gen_imgs)))):
                    try:
                        #gets the image
                        img = gen_imgs[count]
                        #shows the image
                        #reshape to make plt happy
                        axs[row, image].imshow(img.reshape(img.shape[0], img.shape[1]), cmap='gray', vmin=0, vmax=1)
                        #titles with the class
                        axs[row, image].set_title(labels[count], {'fontsize':8})
                        #we don't care about the axis
                        axs[row, image].axis("off")
                        count += 1
                    #to handle when we have ran out of images beacuse they don't fit neatly into even rows
                    except IndexError:
                        axs[row, image].axis("off")
                        
            #full screen style plots
            plt.subplots_adjust(left=0, right=1, bottom=0, top=0.96, wspace=0, hspace=0.7)
            #saves it to the images folder with a name of the current epoch
            figure.savefig(path+"images/"+str(epoch)+".png")
            #closes the figure else it sticks around using up RAM
            plt.close(figure)
            
    epoch += 1
    #increments the progress bar by 1
    progress_bar.update(1)
