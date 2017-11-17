import tensorflow as tf
import numpy as np
import csv
import glob
import skimage.transform

# defines image directory as img, label as csv in label directory
labelNames = "label/labels.csv"
imageNames = "images/*.jpg"
graph = tf.Graph()


def load_data():
    labels = []
    images = []

    # Grabs list of file names for the queue
    files = glob.glob(labelNames)
    # Creates the queue, converts list to tensor
    filename_queue = tf.train.string_input_producer(files)
    # Reader
    reader = tf.WholeFileReader()
    # Returns key and image file, reads the queue into a tensor?
    _, image_file = reader.read(filename_queue)
    # Decodes into tensor
    image = tf.image.decode_jpeg(image_file)
    # Appends tensor to list.
    images.append(image)

    # adds labels from csv to label array, returns array
    with open(labelNames, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            for i in range(len(row)):
                labels.append(int(row[i]))
    return images, labels


def main():
    images, labels = load_data()
    with graph.as_default():

        # take an image list in as 64x64 with 3 color chanels.
        # labels will be 0 or not 1
        images_ph = tf.placeholder(tf.float32, [None, 64, 64, 3])
        labels_ph = tf.placeholder(tf.int32, [None])

        # Flattens image into 1d vector
        images_flat = tf.contrib.layers.flatten(images_ph)

        # input to softmax, starts probabilties. not sure what tf.nn.relu does
        logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)

        # converts labels to tensor
        predicted_labels = tf.argmax(logits, 1)

        # defines loss function, cross entropy. it is better than mean squared.
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels_ph))

        # Adam optimizer is a grad decent optimizer,
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # initalizes variables
        init = tf.global_variables_initializer()

    # Move label and image list to np array
    labels_a = np.array(labels)
    images_a = np.array(images)
    print('this is the np array: ', images_a, '\n')
    print('this won', '\n', images_a.shape, '\n', images_flat, '\n')
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        _, loss_value = sess.run([train, loss], feed_dict={
                                 images_ph: images_a, labels_ph: labels_a})
        im = images_flat.eval()
        print(im)


if __name__ == "__main__":
    main()
