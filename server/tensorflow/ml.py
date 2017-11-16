import tensorflow as tf
import numpy as np
import os
import csv
import glob


# defines image directory as img, label as csv in label directory
data_dir = os.path.join(os.getcwd())
labelNames = 'label/labels.csv'
graph = tf.Graph()


def load_data():
    labels = []
    images = []

    files = glob.glob("images/*.jpg")
    print(files)
    filename_queue = tf.train.string_input_producer(files)
    reader = tf.WholeFileReader()
    _, image_file = reader.read(filename_queue)

    image = tf.image.decode_jpeg(image_file)
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
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

        # Adam optimizer is a grad decent optimizer,
        train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        # initalizes variables
        init = tf.global_variables_initializer()

    labels_a = np.array(labels)
    images_a = np.array(images)
    with tf.Session(graph=graph) as sess:
        sess.run(init)
        for i in range(201):
            _, loss_value = session.run([train, loss], feed_dict={images_ph: images_a, labels_ph: labels_a})

    if i % 10 == 0:
        print("loss value: ", loss_value)


if __name__ == "__main__":
    main()
