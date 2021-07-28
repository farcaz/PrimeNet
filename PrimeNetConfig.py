import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib


def Classifier_loss(loss_object, real_output, caps_output):
    real_loss = loss_object(tf.ones_like(real_output), real_output)
    caps_loss = loss_object(tf.ones_like(caps_output), caps_output)
    total_loss = (real_loss + caps_loss)
    return total_loss


def Feature_loss(loss_object, caps_output):
    return loss_object(tf.ones_like(caps_output), caps_output)


def normalize(x):
    image = tf.cast(x['image'], tf.float32)
    image = (image / 127.5) - 1
    return image

def record_loss(dict, key, value):
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

def save_imgs(epoch, FeatureX,FeatureY,FeatureZ, image):
    #new_image = tf.expand_dims(image,0)

    FeaturesX,fX_imgs = FeatureX(image, training=False)
    FeaturesY,fY_imgs = FeatureY(image, training=False)
    FeaturesZ,fZ_imgs = FeatureZ(image, training=False)


    fig1 = plt.figure(figsize=(4, 4))
    
    

    for i in range(fX_imgs.shape[0]):
        print(i, fX_imgs.shape)
        plt.subplot(4, 4, i +1)
        plt.imshow(fX_imgs[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    fig1.savefig("MNIST_Scale3_%d.png" % epoch)
    plt.clf()
    plt.cla()
    plt.close()

    fig2 = plt.figure(figsize=(4, 4))
    for j in range(fY_imgs.shape[0]):
        plt.subplot(4, 4, j + 1)
        plt.imshow(fY_imgs[j, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig2.savefig("MNIST_Scale5_%d.png" % epoch)
    plt.clf()
    plt.cla()
    plt.close()

    fig3 = plt.figure(figsize=(4, 4))
    for k in range(fZ_imgs.shape[0]):
        plt.subplot(4, 4, k + 1)
        plt.imshow(fZ_imgs[k, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    fig3.savefig("MNIST_Scale7_%d.png" % epoch)
    plt.clf()
    plt.cla()
    plt.close()


def save_loss_plot(epochs,fX_loss_dict,fY_loss_dict,fZ_loss_dict,classifier_loss_dict):
    Fig=plt.figure(figsize=(10,6))
    
    plt.plot(np.arange(0, epochs), fX_loss_dict.values(), label="Layer 3x3")
    plt.plot(np.arange(0, epochs), fY_loss_dict.values(), label="Layer 5x5")
    plt.plot(np.arange(0, epochs), fZ_loss_dict.values(), label="Layer 7x7")
    plt.plot(np.arange(0, epochs), classifier_loss_dict.values(), label="Classifier")
    plt.title("Classifier and shallow network train losses on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig('MNIST_ClassifierVsLayersTrainLoss.png')
    plt.clf()
    plt.cla()
    plt.close()

# bunch of cal per layer
def count_linear(layers):
    MAC = layers.output_shape[1] * layers.input_shape[1]
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[1]
    else:
        ADD = 0
    return MAC*2 + ADD

def count_conv2d(layers, log = False):
    # if log:
    #     print(layers.get_config())
    # number of conv operations = input_h * input_w / stride = output^2
    numshifts = int(layers.output_shape[1] * layers.output_shape[2])
    
    # MAC/convfilter = kernelsize^2 * InputChannels * OutputChannels
    MACperConv = layers.get_config()["kernel_size"][0] * layers.get_config()["kernel_size"][1] * layers.input_shape[3] * layers.output_shape[3]
    
    if layers.get_config()["use_bias"]:
        ADD = layers.output_shape[3]
    else:
        ADD = 0
        
    return MACperConv * numshifts * 2 + ADD

def count_flops(model, log = False):
    '''
    Parameters
    ----------
    model : A keras or TF model
    Returns
    -------
    Sum of all layers FLOPS in unit scale, you can convert it 
    afterward into Millio or Billio FLOPS
    '''

    layer_flops = []
    # run through models
    for layer in model.layers:
        if "dense" in layer.get_config()["name"] or "fc" in layer.get_config()["name"]:
            layer_flops.append(count_linear(layer))
        elif "conv" in layer.get_config()["name"] and "pad" not in layer.get_config()["name"] and "bn" not in layer.get_config()["name"] and "relu" not in layer.get_config()["name"] and "concat" not in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
        elif "res" in layer.get_config()["name"] and "branch" in layer.get_config()["name"]:
            layer_flops.append(count_conv2d(layer,log))
        elif "stage" in layer.get_config()['name']:
            layer_flops.append(count_conv2d(layer,log))
            
    return np.sum(layer_flops)

def get_flops(model):
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=tf.compat.v1.keras.backend.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.


    
