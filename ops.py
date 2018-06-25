import tensorflow as tf
import numpy as np
import os, fnmatch


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD
    
def HSD2RGB(X_HSD):
    
    X_HSD_0, X_HSD_1, X_HSD_2  = tf.split(X_HSD, [1,1,1], axis=3)
    D_R = (X_HSD_1+1) * X_HSD_0
    D_G = 0.5*X_HSD_0*(2-X_HSD_1 + tf.sqrt(tf.constant(3.0))*X_HSD_2)
    D_B = 0.5*X_HSD_0*(2-X_HSD_1 - tf.sqrt(tf.constant(3.0))*X_HSD_2)
    
    #D_R  = tf.expand_dims(D_R,3)
    #D_G  = tf.expand_dims(D_G,3)
    #D_B  = tf.expand_dims(D_B,3)
    
    X_OD = tf.concat([D_R,D_G,D_B],3)
    X_RGB = 1.0 * tf.exp(-X_OD)
    return X_RGB   
    
def HSD2RGB_Numpy(X_HSD):
    
    X_HSD_0 = X_HSD[...,0]
    X_HSD_1 = X_HSD[...,1]
    X_HSD_2 = X_HSD[...,2]
    D_R = np.expand_dims(np.multiply(X_HSD_1+1 , X_HSD_0), 2)
    D_G = np.expand_dims(np.multiply(0.5*X_HSD_0, 2-X_HSD_1 + np.sqrt(3.0)*X_HSD_2), 2)
    D_B = np.expand_dims(np.multiply(0.5*X_HSD_0, 2-X_HSD_1 - np.sqrt(3.0)*X_HSD_2), 2)
     
    X_OD = np.concatenate((D_R,D_G,D_B), axis=2)
    X_RGB = 1.0 * np.exp(-X_OD)
    return X_RGB         
    
def find_files(directory, pattern):
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def image_dist_transform(X, mu, std, pi, Mu_tmpl, Std_tmpl, IMAGE_SIZE, ClusterNo):
    X_conv = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3, ClusterNo))
    for c in range(0, ClusterNo):
        X_norm   = np.divide(np.subtract(np.squeeze(X), mu[c,...]), std[c,...])
        X_univar = np.add(np.multiply(X_norm, Std_tmpl[c,...]), Mu_tmpl[c,...])
        #X_univar = np.add(np.zeros_like(X_norm), mu[c,...])
        X_conv[...,c] = np.multiply(X_univar, np.tile(np.expand_dims(np.squeeze(pi[...,c]), axis=2), (1,1,3)) )
            
    X_conv = np.sum(X_conv, axis=3)
            
    ## Apply the triangular restriction to cxcy plane in HSD color coordinates
    # tf.clip_by_value  ???
    X_conv = np.split(X_conv, 3, axis=2)
    X_conv[1] = np.maximum(np.minimum(X_conv[1], 2.0), -1.0)
    X_conv = np.squeeze(np.swapaxes(np.asarray(X_conv), 0, 3))
            
    ## Transfer from HSD to RGB color coordinates
    X_conv = HSD2RGB_Numpy(X_conv)
    X_conv = np.minimum(X_conv,1.0)
    X_conv = np.maximum(X_conv,0.0)
    
    return X_conv


def weight_variable(shape, stddev=0.02, name=None):
    # print(shape)
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)
        
def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
