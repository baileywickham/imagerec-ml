import tensorflow as tf
import os
import csv

data_dir = os.path.join(os.getcwd(), 'img')
labelNames = 'label/a.csv'

graph = tf.Graph()
def load_data(data_dir):
	labels = []
	images = []	
	with open(labelNames, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			for i in range(len(row)):
				labels.append(int(row[i]))
	filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("./images/*.jpeg"))

	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)

	print('hello')
	print(labels, images)

load_data(data_dir)