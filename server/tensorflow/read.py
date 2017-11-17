import tensorflow as tf
from PIL import Image
import numpy as np
import glob

# Get available files
files = glob.glob("images/*.jpg")
filename_queue = tf.train.string_input_producer(files)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
print(files)
my_img = tf.image.decode_jpeg(value)  # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(len(files)):
    image = my_img.eval()  # Image tensor

    print(image.shape)  # Expecting
  print(image)
  print(my_img)
  #Display image in default image viewer
  #Image.fromarray(np.asarray(image)).show()

  coord.request_stop()
