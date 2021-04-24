# get_image_feature_vectors.py
#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module with Tensorflow
import tensorflow as tf
import tensorflow_hub as hub
# For saving 'feature vectors' into a txt file
import numpy as np
# Glob for reading file names in a folder
import glob
import os.path
import tensorflow_io as tfio
import gc
#################################################
#################################################
# This function:
# Loads the JPEG image at the given path
# Decodes the JPEG image to a uint8 W X H X 3 tensor
# Resizes the image to 224 x 224 x 3 tensor
# Returns the pre processed image as 224 x 224 x 3 tensor
#################################################
def load_img(path):
    # Reads the image file and returns data type of string
    img = tf.io.read_file(path)
    # Decodes the image to W x H x 3 shape tensor with type of uint8
    # expand_animations=False means that it returns 3D array/tensor 
    # even if it's gif(in this case it uses first frame of multiframe images)
    img = tf.io.decode_image(img,channels=3,expand_animations=False)
    
    # Resizes the image to 224 x 224 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)
    # Converts the data type of uint8 to float32 by adding a new axis
    # img becomes 1 x 224 x 224 x 3 tensor with data type of float32
    # This is required for the mobilenet model we are using
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]


    


    return img


#################################################
# This function:
# Loads the mobilenet model in TF.HUB
# Makes an inference for all images stored in a local folder
# Saves each of the feature vectors in a file
#################################################
def get_image_feature_vectors():
    # Definition of module with using tfhub.dev
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    # Loads the module
    module = hub.load(module_handle)
    # Loops through all images in a local folder
    files = glob.glob('..\\copyHunter\\scripts\\images_2\\*')
    # files = glob.glob('..\\copyHunter\\scripts\\images_20k\\*')
    # files.extend(glob.glob('..\\copyHunter\\scripts\\images_50k\\*'))
    # files.extend(glob.glob('..\\copyHunter\\scripts\\images\\*.jpg'))
    for filename in files:
        try:
            print(filename)

            if filename.find('.webp') > 0:
                continue
            # Loads and pre-process the image
            img = load_img(filename)
            # Calculate the image feature vector of the img
            features = module(img)
            # Remove single-dimensional entries from the 'features' array
            feature_set = np.squeeze(features)
           

            # Saves the image feature vectors into a file for later use
            outfile_name = os.path.basename(filename) + ".npz"

            out_path = os.path.join('.\\vectors\\', outfile_name)
            # Saves the 'feature_set' to a text file
            np.savetxt(out_path, feature_set, delimiter=',')
            del img
            del features
            del feature_set
            gc.collect()
        except (Exception) as e:
            print(e)
            continue


get_image_feature_vectors()
