# Typical setup to include TensorFlow.
import tensorflow as tf
from PIL import Image
import numpy as np
# Make a queue of file names including all the JPEG images files in the relative
# image directory.

filename_queue = tf.train.string_input_producer(['./images/a.jpg'])
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  # Start populating the filename queue.

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(1): #length of your filename list
    image = my_img.eval() #here is your image Tensor :) 

  print(image.shape)
  Image.fromarray(np.asarray(image)).show()

  coord.request_stop()