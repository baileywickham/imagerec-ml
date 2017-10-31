import tensorflow as tf
import os
from PIL import Image
files = os.listdir('/home/bailey/workspace/imagerec-ml/img') #make this relational so we can clone it and it will still work without being on just my computer. Lazy hack.
graph = tf.Graph()


#it looks like we are going to need to feed data. where there is a list of file names which get piped into a queue and then
#decoded by a third piece, not sure on how to impliment any of this.

with graph.as_default():

	#take an image list in as 64x64 with 3 color chanels. labels will be andy or not andy
	images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
	labels_ph = tf.placeholder(tf.int8, [None]

	images_flat = tf.contrib.layers.flatten(images_ph)
	
	#input to softmax, starts probabilties. not sure what tf.nn.relu does	
	logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)

