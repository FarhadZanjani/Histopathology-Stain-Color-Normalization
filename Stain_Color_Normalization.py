## Stain-Color Normalization by using Deep Convolutional GMM (DCGMM)
## Developer: VCA group, Eindhoen University of Technology
## Ref: Zanjani, Farhad G., Svitlana Zinger, Babak E. Bejnordi, and Jeroen AWM van der Laak. "Histopathology Stain-Color Normalization Using Deep Generative Models." (2018).


import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc

from model import DCGMM
from config import get_config
from Sample_Provider import SampleProvider
from ops import image_dist_transform

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "prediction", "Mode train/ prediction")
tf.flags.DEFINE_string("logs_dir", "./logs_DGMM_HSD/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/media/farhad/DataVolume1/Data/Pathology_Color_Normalization/StainStudy_Dataset/", "path to dataset")
tf.flags.DEFINE_string("tmpl_dir", "/media/farhad/DataVolume1/Data/Pathology_Color_Normalization/StainStudy_Dataset/Template/", "path to template image(s)")
tf.flags.DEFINE_string("out_dir", "/media/farhad/DataVolume1/Data/Pathology_Color_Normalization/StainStudy_Dataset/output/", "path to template image(s)")


def main():
  sess = tf.Session()
  
  if FLAGS.mode == "train": 
      is_train = True
  else:
      is_train = False
      
  config = get_config(FLAGS, is_train)
  if not os.path.exists(config.logs_dir):
      os.makedirs(config.logs_dir)
    
  dist = DCGMM(sess, config, "DCGMM", is_train)
  db = SampleProvider("H&E_dataset", config, is_train)
  
  if FLAGS.mode == "train":
    
      for i in range(int(config.iteration)):
        X = db.DrawSample(config.batch_size)
        loss = dist.fit(X[0])
        print("iter {:>6d} : {}".format(i+1, loss))
        if i % 500 == 0:
            dist.saver.save(sess, config.logs_dir+ "model.ckpt", i)
        
  elif FLAGS.mode == "prediction":  
    
      if not os.path.exists(config.out_dir):
          os.makedirs(config.out_dir)
     
      db_tmpl = SampleProvider("Template_dataset", config, is_train)
      mu_tmpl = 0
      std_tmpl = 0
      N = 0
      while True:
          X = db_tmpl.DrawSample(config.batch_size)
          if X.size ==0:
              break
          
          mu, std = dist.deploy(X)
          
          N = N+1
          mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
          std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
          
      while True:
          X = db.DrawSample(config.batch_size)
          if X.size ==0:
              break
          
          mu, std, pi = dist.deploy(X)

          X_conv, filename = image_dist_transform(X, mu, std, pi, mu_tmpl, std_tmpl, config.IMAGE_SIZE, config.ClusterNo)
       
          filename = filename.split('/')[-1]
          print(filename)

          if not os.path.exists(config.out_dir):
             os.makedirs(config.out_dir)
          misc.imsave(config.out_dir+filename, np.squeeze(X_conv))        
      
  else:
      print('Invalid "mode" string!')
      return 

if __name__ == "__main__":
  main()
