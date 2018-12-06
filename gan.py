#importing libraries
import numpy as np
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout
from keras import initializers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
import keras

#initialization
input_size = 10
batch_size = 50
epoch_size = 10
features= X_train[0]
length = X_train

#generator network
def create_generator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = input_size,output_dim=100,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    
    #internal layers
    classifier.add(Dense(output_dim =32))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dense(output_dim = 32))
    classifier.add(LeakyReLU(0.2))
    
    #output layer
    classifier.add(Dense(output_dim =features, activation = 'tanh'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier

generator = create_generator()

#discriminator network
def create_discriminator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim =features , output_dim = 10,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.5))
    
    #internal layers
    classifier.add(Dense(output_dim =20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))

    #output layer
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier 

discriminator = create_discriminator()

#gan network
def create_gan_network():

    discriminator.trainable = False
    gan_input = Input(shape=(input_size,))
    
    x = generator(gan_input)
    gan_output = discriminator(x)
    
    #model network gan    
    gan = Model(inputs=gan_input, outputs=gan_output)
    
    #compiling
    gan.compile(loss='binary_crossentropy',optimizer = keras.optimizers.Adam(lr=0.001,amsgrad=True))
    return gan

gan = create_gan_network()

#training gan
def train_gan():
    
   for epoch in range(1,epoch_size+1):
        
       print('--epoch : %d--' %epoch)
       j=0
       
       for i in range(batch_size,length,batch_size):
            
            #discriminator training
            X_curr = X_train[j:i,:]
            noise = np.random.normal(0, 1, size=[batch_size,input_size])
            generated_output = generator.predict(noise)
        
            X_dis = np.concatenate([X_curr, generated_output])
            y_dis = np.zeros(len(X_dis))
            y_dis[:batch_size]=0.9
            
            discriminator.trainable = True
            discriminator.train_on_batch(X_dis,y_dis)
            
            #generator training
            discriminator.trainable = False
            gan_input = np.random.normal(0, 1, size=[batch_size,input_size])
            gan_output = np.ones(batch_size)
            gan.train_on_batch(gan_input, gan_output)
            
            j=i

train_gan()

#generated data
sample_size=100
generated = generator.predict(np.random.normal(0, 1, size=[sample_size,input_size]))

