class Config(object):
  pass

def get_config(FLAGS, is_train):
    
  config = Config()

  config.logs_dir = FLAGS.logs_dir
  config.data_dir = FLAGS.data_dir
  config.mode     = FLAGS.mode

  config.ClusterNo= 4  
  config.batch_size = 1  # The current implementation only supports a batch size equal to unity!

  if is_train:
    config.im_size = 512 # The width and height should be equal. Upper bound of the input image size is limited to the GPU memory.
    config.lr = 1e-4
    config.iteration = 10e6

    config.image_options = {'resize': False, 'resize_size': config.im_size, 'crop': True, 'flip': True, 'rotate_stepwise': True}
    config.fileformat = 'tif'
    config.tmp_dir = "tmp"
    config.ckpt_dir = "ckpt"
    
  else:
    config.batch_size = 1
    config.im_size = 512 # The width and height should be equal. Upper bound of the input image size is limited to the GPU memory.

    config.image_options = {'resize': False, 'resize_size': config.im_size, 'crop': True, 'flip': False, 'rotate_stepwise': False}
    config.fileformat = 'tif'
    config.result_dir = "result"
    config.ckpt_dir = "ckpt"
  return config
