import itertools
import math

import numpy as np
import tensorflow as tf
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from src.data import image_augmentation
from src.data import emnist


class Dataset:
    def __init__(self, x_train, y_train, batch_size=50, x_test=None, y_test=None):
        self.batch_size = batch_size

        # LIMIT = 5000 # for debugging remove this
        # x_train, y_train = x_train[:LIMIT], y_train[:LIMIT]

        self.x_train, self.y_train = self.shuffle(x_train, y_train)
        self.x_test, self.y_test = x_test, y_test
        self.x_aux, self.y_aux, self.mal_aux_labels = None, None, None
        self.x_aux_test, self.mal_aux_labels_test = None, None

        self.fg = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))

    def shuffle(self, x, y):
        perms = np.random.permutation(x.shape[0])
        return x[perms, :], y[perms]

    def get_data(self):
        """Creates one batch of data.
        # This is a TERRIBLE way to load data... every epoch we get it in the same order !!!

        Yields:˚
            tuple of two: input data batch and corresponding labels
        """
        # count = int(self.x_train.shape[0] / self.batch_size)
        # if count == 0:
        #     yield self.x_train, self.y_train
        #     # return [(self.x_train, self.y_train)]
        # for bid in range(count): # Note: Unsafe if batch_size is small!!!
        #     batch_x = self.x_train[bid * self.batch_size:(bid + 1) * self.batch_size]
        #     batch_y = self.y_train[bid * self.batch_size:(bid + 1) * self.batch_size]
        #
        #     yield batch_x, batch_y
        # bid = 0

        shuffle_size = min(self.x_train.shape[0], 10000)

        return self.fg \
            .shuffle(shuffle_size) \
            .batch(self.batch_size, drop_remainder=True) \
            .prefetch(tf.data.experimental.AUTOTUNE)

    def get_aux(self, mal_num_batch):
        """Creates one batch of data.

        Yields:
            tuple of two: input data batch and corresponding labels
        """
        if int(self.x_aux.shape[0] / self.batch_size) < 1:
            yield self.x_aux, self.mal_aux_labels

        for bid in range(int(self.x_aux.shape[0] / self.batch_size)):
            batch_x = self.x_aux[bid * self.batch_size:(bid + 1) * self.batch_size]
            batch_y = self.mal_aux_labels[bid * self.batch_size:(bid + 1) * self.batch_size]

            yield batch_x, batch_y
        bid = 0

    def get_data_with_aux(self, insert_aux_times, num_batches, pixel_pattern=None, noise_level=None):
        """Creates one batch of data with the AUX data inserted `insert_aux_times` per batch with malicious labels.

        :param insert_aux_times number of times aux should be inserted per batch. 1 for Bagdasaryan
        :param num_batches number of batches to generate. 200 for Bagdasaryan
        :param noise_level sigma of normal distribution noise to add to training samples

        Yields:
            tuple of two: input data batch and corresponding labels
        """

        multiplier = max(float(insert_aux_times) / float(self.mal_aux_labels.shape[0]),
                         1)  # potential multiplier if aux is smaller than insert
        number_of_mal_items = int(multiplier * num_batches)

        r1 = insert_aux_times
        r2 = self.batch_size - insert_aux_times

        normal_mult = max(float(num_batches) * float(self.batch_size) / self.x_train.shape[0], 1)

        normal_fg = self.fg \
            .repeat(int(math.ceil(normal_mult * self.x_train.shape[0]))) \
            .shuffle(self.x_train.shape[0]) \
            .batch(r2, drop_remainder=True) \

        if insert_aux_times == 0:
            return normal_fg

        mal_fb = tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels)) \
            .repeat(number_of_mal_items) \
            .shuffle(number_of_mal_items)

        if noise_level is not None: # Add noise
            mal_fb = mal_fb.map(image_augmentation.add_noise(noise_level))

        mal_fb = mal_fb.batch(r1, drop_remainder=True)

        zipped = tf.data.Dataset.zip((mal_fb, normal_fg)).map(lambda x, y:
                                                              (tf.concat((x[0], y[0]), axis=0),
                                                               tf.concat((x[1], y[1]), axis=0))
                                                              )
        result = zipped.unbatch()
        return result.batch(self.batch_size, drop_remainder=True) \
            .take(num_batches)

    def get_aux_test_generator(self, aux_size):
        if aux_size == 0:
            return tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.experimental.AUTOTUNE)

        return tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
            .repeat(aux_size) \
            .batch(self.batch_size, drop_remainder=False) \
            .prefetch(tf.data.experimental.AUTOTUNE)


    @staticmethod
    def keep_samples(x_train, y_train, number_of_samples):
        if number_of_samples == -1:
            return x_train, y_train

        perms = np.random.permutation(number_of_samples)
        return x_train[perms, :], y_train[perms]

    @staticmethod
    def keep_samples_iterative(x_train, y_train, number_of_samples):
        if number_of_samples == -1:
            return x_train, y_train

        perms = [np.random.permutation(min(number_of_samples, val.shape[0])) for val in x_train]

        return [val[perm, :] for val, perm in zip(x_train, perms)], \
               [val[perm] for val, perm in zip(y_train, perms)]

    @staticmethod
    def apply_trigger(x_aux):
        triggersize = 4
        trigger = np.ones((x_aux.shape[0], triggersize, triggersize, 1))
        out = x_aux
        out[:, 0:triggersize, 0:triggersize, :] = trigger
        return out


    @staticmethod
    def get_mnist_dataset(number_of_samples):
        """MNIST dataset loader"""
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_fmnist_dataset(number_of_samples):
        """Fashion MNIST dataset loader"""
        fmnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fmnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
        x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)
        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_cifar10_dataset(number_of_samples):
        """Cifar10 dataset loader"""
        cifar = tf.keras.datasets.cifar10
        (x_train, y_train), (x_test, y_test) = cifar.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)

        y_train, y_test = np.squeeze(y_train, axis=1), np.squeeze(y_test, axis=1)

        x_train, y_train = Dataset.keep_samples(x_train, y_train, number_of_samples)
        x_test, y_test = Dataset.keep_samples(x_test, y_test, -1) # Note: hardcoded

        # Subtract
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

        return (x_train, y_train), (x_test, y_test)

    @staticmethod
    def get_emnist_dataset(number_of_samples, number_of_clients, normalize_mnist_data):
        """nonIID MNIST dataset loader"""
        train_dataset, test_dataset = emnist.load_data()
        x_train, y_train = np.array([1.0 - np.array(val['pixels']) for val in train_dataset]), \
                           np.array([np.array(val['label']).astype(np.uint8) for val in train_dataset])
        x_test, y_test = np.array([1.0 - np.array(val['pixels']) for val in test_dataset]), \
                           np.array([np.array(val['label']).astype(np.uint8) for val in test_dataset])

        if normalize_mnist_data:
            emnist_mean, emnist_std = 0.036910772, 0.16115953
            x_train = np.array([(x - emnist_mean) / emnist_std for x in x_train])
            x_test = np.array([(x - emnist_mean) / emnist_std for x in x_test])

        # Randomly assign clients to buckets but keep them as client
        if number_of_clients < x_train.shape[0]:
            assignments = np.random.randint(0, number_of_clients, x_train.shape[0], dtype=np.uint16)
            new_x_train = []
            new_y_train = []
            new_x_test = []
            new_y_test = []
            for i in range(number_of_clients):
                new_x_train.append(
                    np.concatenate(x_train[assignments == i], axis=0)
                )
                new_y_train.append(
                    np.concatenate(y_train[assignments == i], axis=0)
                )
                new_x_test.append(
                    np.concatenate(x_test[assignments == i], axis=0)
                )
                new_y_test.append(
                    np.concatenate(y_test[assignments == i], axis=0)
                )
            #
            # new_x_train = np.concatenate(new_x_train, axis=0)
            # new_y_train = np.concatenate(new_y_train, axis=0)
            # new_x_test = np.concatenate(new_x_test, axis=0)
            # new_y_test = np.concatenate(new_y_test, axis=0)

            if number_of_samples == -1:
                number_of_samples_per_client = -1
            else:
                number_of_samples_per_client = int(number_of_samples / float(number_of_clients))

            x_train, y_train = Dataset.keep_samples_iterative(new_x_train, new_y_train, number_of_samples_per_client)
            x_test, y_test = Dataset.keep_samples_iterative(new_x_test, new_y_test,
                                                            min(number_of_samples_per_client, 500))
        elif number_of_clients > x_train.shape[0]:
            print(f"Number of clients {number_of_clients} is large than amount of EMNIST users {x_train.shape[0]}")
        else:
            print("Exactly using EMNIST as clients!")

        x_train, x_test = [val.astype(np.float32)[..., np.newaxis] for val in x_train], \
                          [val.astype(np.float32)[..., np.newaxis] for val in x_test]

        return (x_train, y_train), (x_test, y_test)


class ImageGeneratorDataset(Dataset):
    def __init__(self, x_train, y_train, batch_size=50, x_test=None, y_test=None):
        super().__init__(x_train, y_train, batch_size, x_test, y_test)

    def get_aux_test_generator(self, aux_size):
        if aux_size == 0:
            return tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
                .batch(self.batch_size, drop_remainder=False) \
                .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
            .repeat(aux_size) \
            .batch(self.batch_size, drop_remainder=False) \

        return test_dataset \
            .map(image_augmentation.test_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

    def get_data(self):
        return self.fg\
            .shuffle(self.x_train.shape[0]) \
            .batch(self.batch_size, drop_remainder=True) \
            .map(image_augmentation.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .prefetch(tf.data.experimental.AUTOTUNE)

    def get_aux(self, mal_num_batch):
        multiplier = max(float(self.batch_size) / float(self.mal_aux_labels.shape[0]),
                         1)  # potential multiplier if aux is smaller than insert
        number_of_mal_items = int(multiplier * mal_num_batch)
        return tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels)) \
            .repeat(number_of_mal_items) \
            .batch(self.batch_size, drop_remainder=False) \
            .map(image_augmentation.train_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \


    def get_data_with_aux(self, insert_aux_times, num_batches, noise_level=None):
        """Creates one batch of data with the AUX data inserted `insert_aux_times` per batch with malicious labels.

        :param insert_aux_times number of times aux should be inserted per batch. 1 for Bagdasaryan
        :param num_batches number of batches to generate. 200 for Bagdasaryan

        Yields:
            tuple of two: input data batch and corresponding labels
        """
        # assert self.y_aux != [] and self.x_aux != [] and self.mal_aux_labels != []

        multiplier = float(insert_aux_times) / float(self.mal_aux_labels.shape[0]) # potential multiplier if aux is smaller than insert
        number_of_mal_items = int(math.ceil(multiplier * num_batches))


        r1 = insert_aux_times
        r2 = self.batch_size - insert_aux_times

        normal_mult = max(float(num_batches) * float(self.batch_size) / self.x_train.shape[0], 1)

        normal_fg = self.fg\
            .repeat(int(math.ceil(normal_mult))) \
            .shuffle(self.x_train.shape[0]) \
            .batch(r2, drop_remainder=True) \
            .map(image_augmentation.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \

        if insert_aux_times == 0:
            print(f"Insert 0 {normal_mult}")
            return normal_fg

        mal_fb = tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels)) \
            .repeat(number_of_mal_items) \
            .shuffle(number_of_mal_items * self.mal_aux_labels.shape[0])

        mal_fb = mal_fb.batch(r1, drop_remainder=True) \
            .map(image_augmentation.train_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \

        if noise_level is not None:  # Add noise
            mal_fb = mal_fb.map(image_augmentation.add_noise_batch(noise_level))

        zipped = tf.data.Dataset.zip((mal_fb, normal_fg)).map(lambda x, y:
                                                              (tf.concat((x[0], y[0]), axis=0), tf.concat((x[1], y[1]), axis=0))
                                                              )
        result = zipped.unbatch()
        return result.batch(self.batch_size, drop_remainder=True)\
            .take(num_batches)


class PixelPatternDataset(ImageGeneratorDataset):

    def __init__(self, x_train, y_train, target_label, batch_size=50, x_test=None, y_test=None):
        super().__init__(x_train, y_train, batch_size, x_test, y_test)

        # Assign train set part
        (self.x_aux, self.y_aux) = \
            (self.x_train, self.y_train)
        self.mal_aux_labels = np.repeat(target_label, self.y_aux.shape).astype(np.uint8)

        self.pixel_pattern = 'basic'

    def get_aux_test_generator(self, aux_size):
        if aux_size == 0:
            return tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
                .batch(self.batch_size, drop_remainder=False) \
                .map(image_augmentation.add_pixel_pattern(self.pixel_pattern)) \
                .prefetch(tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_aux_test, self.mal_aux_labels_test)) \
            .repeat(aux_size) \
            .batch(self.batch_size, drop_remainder=False) \

        return test_dataset \
            .map(image_augmentation.test_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
            .map(image_augmentation.add_pixel_pattern(self.pixel_pattern)) \
            .prefetch(tf.data.experimental.AUTOTUNE)


    def get_data_with_aux(self, insert_aux_times, num_batches, noise_level=None):
        """Creates one batch of data with the AUX data inserted `insert_aux_times` per batch with malicious labels.

        :param insert_aux_times number of times aux should be inserted per batch. 1 for Bagdasaryan
        :param num_batches number of batches to generate. 200 for Bagdasaryan

        Yields:
            tuple of two: input data batch and corresponding labels
        """
        # assert self.y_aux != [] and self.x_aux != [] and self.mal_aux_labels != []

        multiplier = float(insert_aux_times) / float(
            self.mal_aux_labels.shape[0])  # potential multiplier if aux is smaller than insert
        number_of_mal_items = int(math.ceil(multiplier * num_batches))

        r1 = insert_aux_times
        r2 = self.batch_size - insert_aux_times

        normal_mult = max(float(num_batches) * float(self.batch_size) / self.x_train.shape[0], 1)

        normal_fg = self.fg \
            .repeat(int(math.ceil(normal_mult))) \
            .shuffle(self.x_train.shape[0]) \
            .batch(r2, drop_remainder=True) \
            .map(image_augmentation.augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if insert_aux_times == 0:
            return normal_fg

        mal_fb = tf.data.Dataset.from_tensor_slices((self.x_aux, self.mal_aux_labels)) \
            .repeat(number_of_mal_items) \
            .shuffle(number_of_mal_items * self.mal_aux_labels.shape[0])

        mal_fb = mal_fb.batch(r1, drop_remainder=True) \
            .map(image_augmentation.train_aux_augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if noise_level is not None:  # Add noise
            mal_fb = mal_fb.map(image_augmentation.add_noise_batch(noise_level))

        mal_fb = mal_fb.map(image_augmentation.add_pixel_pattern(self.pixel_pattern))

        zipped = tf.data.Dataset.zip((mal_fb, normal_fg)).map(lambda x, y:
                                                              (tf.concat((x[0], y[0]), axis=0),
                                                               tf.concat((x[1], y[1]), axis=0))
                                                              )
        result = zipped.unbatch()
        return result.batch(self.batch_size, drop_remainder=True)\
            .take(num_batches)


class GeneratorDataset(Dataset):

    def __init__(self, generator, batch_size):
        super().__init__([], [], 0, None, None)
        self.generator = generator
        self.batch_size = batch_size

    def get_data(self):
        return self.generator\
            .batch(self.batch_size)\
            .prefetch(tf.data.experimental.AUTOTUNE)

