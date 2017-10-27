import tensorflow as tf
import os
from PIL import  Image


image = tf.image.decode_jpeg(tf.read_file("a.jpg"), channels=3)
re_im = tf.image.resize_images(image, [299, 299])
bw = tf.image.rgb_to_grayscale(re_im)
with  tf.Session() as sess:
    print(sess.run(bw))

