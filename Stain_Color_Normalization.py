''' * Stain-Color Normalization by using Deep Convolutional GMM (DCGMM).
    * VCA group, Eindhoen University of Technology.
    * Ref: Zanjani F.G., Zinger S., Bejnordi B.E., van der Laak J. AWM, de With P. H.N., "Histopathology Stain-Color Normalization Using Deep Generative Models", (2018).'''


import tensorflow as tf
import numpy as np
import os
import scipy.misc as misc

from model import DCGMM
from config import get_config
from Sample_Provider import SampleProvider
from ops import image_dist_transform
import ops as utils

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('mode', "prediction", "Mode train/ prediction")
tf.flags.DEFINE_string("logs_dir", "./logs_DGMM_HSD/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/media/farhad/DataVolume1/Data/Pathology_Color_Normalization/StainStudy_Dataset/", "path to dataset")
tf.flags.DEFINE_string("tmpl_dir", None, "path to template image(s)")
tf.flags.DEFINE_string("out_dir", None, "path to template image(s)")


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
  db = SampleProvider("Train_dataset", config.data_dir, config.fileformat, config.image_options, is_train)
  
  if FLAGS.mode == "train":
      
      for i in range(int(config.iteration)):
        X = db.DrawSample(config.batch_size)
        X_hsd = utils.RGB2HSD(X[0]/255.0)
        loss, summary_str, summary_writer = dist.fit(X_hsd)
        
        if i % config.ReportInterval == 0:
            summary_writer.add_summary(summary_str, i)
            print("iter {:>6d} : {}".format(i+1, loss))
            
        if i % config.SavingInterval == 0:
            dist.saver.save(sess, config.logs_dir+ "model.ckpt", i)
        
  elif FLAGS.mode == "prediction":  
    
      if not os.path.exists(config.out_dir):
          os.makedirs(config.out_dir)
     
      db_tmpl = SampleProvider("Template_dataset", config.tmpl_dir, config.fileformat, config.image_options, is_train)
      mu_tmpl = 0
      std_tmpl = 0
      N = 0
      while True:
          X = db_tmpl.DrawSample(config.batch_size)
          
          if len(X) ==0:
              break
          
          X_hsd = utils.RGB2HSD(X[0]/255.0)
          
          mu, std, gamma = dist.deploy(X_hsd)
          mu = np.asarray(mu)
          mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
          std = np.asarray(std)
          std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]
          
          N = N+1
          mu_tmpl  = (N-1)/N * mu_tmpl + 1/N* mu
          std_tmpl  = (N-1)/N * std_tmpl + 1/N* std
      
      print("Estimated Mu for template(s):")
      print(mu_tmpl)
      
      print("Estimated Sigma for template(s):")
      print(std_tmpl)
      
      db = SampleProvider("Test_dataset", config.data_dir, config.fileformat, config.image_options, is_train)
      while True:
          X = db.DrawSample(config.batch_size)
          
          if len(X) ==0:
              break
          
          X_hsd = utils.RGB2HSD(X[0]/255.0)
          mu, std, pi = dist.deploy(X_hsd)
          mu = np.asarray(mu)
          mu  = np.swapaxes(mu,1,2)   # -> dim: [ClustrNo x 1 x 3]
          std = np.asarray(std)
          std  = np.swapaxes(std,1,2)   # -> dim: [ClustrNo x 1 x 3]

          X_conv = image_dist_transform(X_hsd, mu, std, pi, mu_tmpl, std_tmpl, config.im_size, config.ClusterNo)
       
          filename = X[1]
          filename = filename[0].split('/')[-1]
          print(filename)

          if not os.path.exists(config.out_dir):
             os.makedirs(config.out_dir)
          misc.imsave(config.out_dir+filename, np.squeeze(X_conv))        
      
  else:
      print('Invalid "mode" string!')
      return 

if __name__ == "__main__":
  main()
