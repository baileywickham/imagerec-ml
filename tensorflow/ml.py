import tensorflow as tf
import os
from PIL import Image

labels = []
images = []

directories = [d for d in os.listdir(data_dir)
		if os.path.isdir(os.path.join(data_dir, d))]

for d in directories
	label_dir = os.path.join(data_dir, d)
	file_names = [os.path.join(label_dir, f)
			for f in os.listdir(label_dir)
			if f.endswith(".jpg")]
	for f in file_names:
		images.append(skimage.data.read(f))
		labels.append(int(d))
images, labels = load_data(train_data_dir)

#it looks like we are going to need to feed data. where there is a list of file names which get piped into a queue and then
#decoded by a third piece, not sure on how to impliment any of this.

with graph.as_default():

	#take an image list in as 64x64 with 3 color chanels. labels will be andy or not andy
	images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
	labels_ph = tf.placeholder(tf.int8, [None]

	images_flat = tf.contrib.layers.flatten(images_ph)
	
	#input to softmax, starts probabilties. not sure what tf.nn.relu does	
	logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)

