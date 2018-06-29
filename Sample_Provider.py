import numpy as np
import scipy.misc as misc
from scipy.ndimage import rotate
from ops import find_files
import os


class SampleProvider(object):
  
  def __init__(self, name, data_dir, fileformat, image_options, is_train):
    self.name = name
    self.is_train = is_train
    self.path = data_dir
    self.fileformat = fileformat
    self.reset_batch_offset()
    self.files = self._create_image_lists()
    self.image_options = image_options
    self._read_images()
    
  def _create_image_lists(self):
    if not os.path.exists(self.path):    
        print("Image directory '" + self.path + "' not found.")
        return None
    
    file_list = list()

    for filename in find_files(self.path, '*.' + self.fileformat):
        file_list.append(filename)

    print ('No. of files: %d' % (len(file_list)))
    return file_list

  def _read_images(self):
    self.__channels = True
    self.images_org = np.array([misc.imread(filename) for filename in self.files])
   
  def _transform(self, images_org):
        
    if self.image_options["crop"]:
        resize_size = int(self.image_options["resize_size"])
        y  = np.random.permutation(range(resize_size/2, images_org.shape[0]-resize_size/2))
        y = int(y[0])
        y1 = int(y - resize_size/2.0)
        y2 = int(y + resize_size/2.0)
        
        x  = np.random.permutation(range(resize_size/2, images_org.shape[1]-resize_size/2))
        x = int(x[0])
        x1 = int(x - resize_size/2.0)
        x2 = int(x + resize_size/2.0)
        
        image = images_org[y1:y2, x1:x2,...]
        
    if self.image_options["resize"]:
        resize_size = int(self.image_options["resize_size"])
        image = misc.imresize(image, [resize_size, resize_size], interp='nearest')
        
    if self.image_options["flip"]:
        if(np.random.rand()<.5):
            image = image[::-1,...]
            
        if(np.random.rand()<.5):
            image = image[:,::-1,...]
            
    if self.image_options["rotate_stepwise"]:
            if(np.random.rand()>.25): # skip "0" angle rotation
                angle = int(np.random.permutation([1,2,3])[0] * 90)
                image = rotate(image, angle, reshape=False)

    return np.array(image)

  def get_records(self):
    return self.files, self.annotations
        
  def get_records_info(self):
      return self.files
        
  def reset_batch_offset(self, offset=0):
      self.batch_offset = offset
      self.epochs_completed = 0

  def DrawSample(self, batch_size):
    start = self.batch_offset
    self.batch_offset += batch_size
    if self.batch_offset > self.images_org.shape[0]:
        
        if not self.is_train:
            image = []
            return image
            
        # Finished epoch
        self.epochs_completed += 1
        print(">> Epochs completed: #" + str(self.epochs_completed))
        # Shuffle the data
        perm = np.arange(self.images_org.shape[0], dtype=np.int)
        np.random.shuffle(perm)
        
        self.images_org = self.images_org[perm]
        self.files = [self.files[k] for k in perm] 
        
        # Start next epoch
        start = 0
        self.batch_offset = batch_size

    end = self.batch_offset
    
  
    image = [self._transform(self.images_org[k]) for k in range(start,end)]
    curfilename = [self.files[k] for k in range(start,end)] 
    
    return np.asarray(image), curfilename
    

