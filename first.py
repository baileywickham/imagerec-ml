import tensorflow as tf
import os
from PIL import Image
files = os.listdir('/home/bailey/workspace/imagerec-ml/img')
#make this relational so we can clone it and it will still work without being on just my computer. Lazy hack.

if __name__ == '__main__':
    for file in files:
        print(file)
        bw = tf.image.rgb_to_grayscale(tf.image.resize_images(tf.image.decode_jpeg(tf.read_file(file), channels=3), [299, 299]))
        #this doesnt work and isn't the propper way to do it but its staying until I or andy gets around to fixing it
with tf.Session() as sess:
    print(sess.run(bw))
    file = open("temp", 'w')
    file.write(sess.run(bw))
