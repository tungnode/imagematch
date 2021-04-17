import numpy as np
import threading
import glob
import tensorflow as tf

dataset = tf.data.Dataset.range(8) 
dataset = dataset.batch(3)
for x in dataset:
    print(x)