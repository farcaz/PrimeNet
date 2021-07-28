import os
import time

import tensorflow as tf  # TF 2.0
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist, fashion_mnist,cifar10
from extra_keras_datasets import stl10
#from model import Generator, Discriminator
from PrimeNetModel import Capsule, PrimeNet
from PrimeNetConfig import Feature_loss, Classifier_loss, save_imgs, normalize, record_loss,save_loss_plot,count_flops,get_flops
import matplotlib as plot
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten

from sklearn.manifold import TSNE
from keras.datasets import mnist
from sklearn.datasets import load_iris
from numpy import reshape
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model_profiler import model_profiler
from tensorflow.python.client import device_lib
import keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from sklearn.cluster import KMeans
#from sklearn.metrics import fowlkes_mallows
#from sklearn.metrics.cluster import adjusted_mutual_info_score

#Classifier = PrimeNet()

def train():
    data, info = tfds.load("mnist", with_info=True, data_dir='tensorflow_datasets')
    train_data = data['train']

    if not os.path.exists('./images/mnist'):
        os.makedirs('./images/mnist')

    # settting hyperparameter
    latent_dim = 100
    epochs = 100
    batch_size = 200
    buffer_size = 6000
    save_interval = 10

    FeatureX = Capsule(32,64,3)
    Classifier = PrimeNet()
    FeatureY = Capsule(32,64,5)
    FeatureZ = Capsule(32,64,7)
    #print(generator.summary(),generator2.summary(),generator3.summary(),discriminator.summary())

    optimizer = tf.keras.optimizers.Adam(0.0002, 0.99)
    
    train_dataset = train_data.map(normalize).shuffle(buffer_size).batch(batch_size)

    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    
    def feature_train(images):
        

        with tf.GradientTape(persistent=True) as tape:
            #Multi-scale feature and image representations
            FeatureX_Output,ImageX=FeatureX(images)
            FeatureY_Output,ImageY=FeatureY(images)
            FeatureZ_Output,ImageZ=FeatureZ(images)


           
            FeatureX_loss = Feature_loss(cross_entropy, FeatureX_Output)
            FeatureY_loss = Feature_loss(cross_entropy, FeatureY_Output)
            FeatureZ_loss = Feature_loss(cross_entropy, FeatureZ_Output)



            if((FeatureX_loss<FeatureY_loss)&(FeatureX_loss<FeatureZ_loss)):
                classifier_output = Classifier(FeatureX_Output)
                classifier_loss = Classifier_loss(cross_entropy, classifier_output, FeatureX_Output)
                print("Selected Loss: Conv3x3 ",FeatureX_loss)
            elif((FeatureY_loss<FeatureX_loss)&(FeatureY_loss<FeatureZ_loss)):
                classifier_output = Classifier(FeatureY_Output)
                classifier_loss = Classifier_loss(cross_entropy, classifier_output, FeatureY_Output)
                print("Selected Loss: Conv5x5 ",FeatureY_loss)
            else:
                classifier_output = Classifier(FeatureZ_Output)
                classifier_loss = Classifier_loss(cross_entropy, classifier_output, FeatureZ_Output)
                print("Selected Loss: Conv7x7 ",FeatureZ_loss)


           
                    
        grad_FeatureX = tape.gradient(FeatureX_loss, FeatureX.trainable_variables)
        optimizer.apply_gradients(zip(grad_FeatureX, FeatureX.trainable_variables))

        grad_FeatureY = tape.gradient(FeatureY_loss, FeatureY.trainable_variables)
        optimizer.apply_gradients(zip(grad_FeatureY, FeatureY.trainable_variables))

        grad_FeatureZ = tape.gradient(FeatureZ_loss, FeatureZ.trainable_variables)
        optimizer.apply_gradients(zip(grad_FeatureZ, FeatureZ.trainable_variables))

        grad_classifier = tape.gradient(classifier_loss, Classifier.trainable_variables)
        optimizer.apply_gradients(zip(grad_classifier, Classifier.trainable_variables))
        
        
        return FeatureX_loss,FeatureY_loss,FeatureZ_loss, classifier_loss

    seed = tf.random.normal([16, latent_dim])
    print(seed)

    FeatureX_loss_dict={}
    FeatureY_loss_dict={}
    FeatureZ_loss_dict={}
    Classifier_loss_dict={}
    
    for epoch in range(epochs):

        start = time.time()
        total_featureX_loss = 0
        total_featureY_loss = 0
        total_featureZ_loss = 0

        total_classifier_loss = 0
        #total_disc5_loss = 0
        #total_disc7_loss = 0

        for images in train_dataset:
            featureX_loss,featureY_loss,featureZ_loss, classifier_loss = feature_train(images)
            print('Loss at per input batch by features at scale 3 = {}, scale 5  = {}, scale 7 = {}, Classifier  ={}'.format(featureX_loss , featureY_loss, featureZ_loss,classifier_loss))
            
            total_featureX_loss += featureX_loss
            total_featureY_loss += featureY_loss
            total_featureZ_loss += featureZ_loss
            
            total_classifier_loss += classifier_loss
            #total_disc5_loss += disc5_loss
            #total_disc7_loss += disc7_loss

        record_loss(FeatureX_loss_dict,epoch,total_featureX_loss/ batch_size)
        record_loss(FeatureY_loss_dict,epoch,total_featureY_loss/ batch_size)
        record_loss(FeatureZ_loss_dict,epoch,total_featureZ_loss/ batch_size)
        record_loss(Classifier_loss_dict,epoch,total_classifier_loss/ batch_size)
            

        print('Time for epoch {} is {} sec - FeatureX loss = {}, FeatureY loss = {},FeatureZ loss = {},  Model loss = {}'.format(epoch + 1, time.time() - start, total_featureX_loss / batch_size,total_featureY_loss / batch_size,total_featureZ_loss / batch_size, total_classifier_loss / batch_size))
        if epoch % save_interval == 0:
                print(images.shape)
                save_imgs(epoch, FeatureX,FeatureY,FeatureZ, images[0:16])
    save_loss_plot(epochs,FeatureX_loss_dict,FeatureY_loss_dict,FeatureZ_loss_dict,Classifier_loss_dict)
  


if __name__ == "__main__":
    train()
