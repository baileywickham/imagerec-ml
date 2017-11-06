import tensorflow as tf
import os
from PIL import Image

data_dir = os.path.join(os.getcwd(), 'img')
def load_data(data_dir):
	labels = []
	images = []

	#data_dir define data directory eventally
	directories = [d for d in os.listdir(data_dir)
			if os.path.isdir(os.path.join(data_dir, d))]

	for d in directories:
		label_dir = os.path.join(data_dir, d)
		file_names = [os.path.join(label_dir, f)
				for f in os.listdir(label_dir)
				if f.endswith(".jpg")]
		for f in file_names:
			images.append(skimage.data.read(f))
			labels.append(int(d))
	return images, labels


images, labels = load_data(data_dir)

#it looks like we are going to need to feed data. where there is a list of file names which get piped into a queue and then
#decoded by a third piece, not sure on how to impliment any of this.
graph = tf.Graph()

with graph.as_default():

	#take an image list in as 64x64 with 3 color chanels. labels will be andy or not andy
	images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
	labels_ph = tf.placeholder(tf.int8, [None])
	
	#Flattens image into 1d vector
	images_flat = tf.contrib.layers.flatten(images_ph)
	
	#input to softmax, starts probabilties. not sure what tf.nn.relu does	
	logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)
	#converts labels to tensor
	predicted_labels = tf.argmax(logits, 1)
	
	#defines loss function, cross entropy. For some reason it is better than mean squared.
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels_ph))
	
	#Adam optimizer is a grad decent optimizer, telling it to minimize the loss function defined here.
	train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	
	#initalizes variables
	init = tf.global_variables_initializer()

#sets session variable, inits vars
session = tf.Session(graph=graph)
session.run(init)	

for i in range(201):
	_, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})
	
	if i % 10 == 0:
		print("loss value: ", loss_value)
