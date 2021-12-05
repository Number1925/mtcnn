import cv2
import tensorflow as tf
import numpy as np
from tensorflow._api.v2 import image


def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs, max_delta=0.2)
    inputs = tf.image.random_saturation(inputs, lower=0.5, upper=1.5)
    return inputs


def read_tf(imgs, net_size, batch_size):
    # 读取tfrecord文件 并进行shuffle
    raw_image_dataset = tf.data.TFRecordDataset(imgs).shuffle(2000)

    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/roi': tf.io.FixedLenFeature([4], tf.float32),
        'image/landmark': tf.io.FixedLenFeature([10], tf.float32)
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    # 进行图片部分的操作
    image_batch = parsed_image_dataset.map(
        lambda img_feature: tf.io.decode_raw(img_feature['image/encoded'], tf.uint8))

    image_batch = image_batch.map(
        lambda image_feature:  tf.reshape(image_feature, [net_size, net_size, 3]))
    image_batch = image_batch.map(
        lambda image_feature: (tf.cast(image_feature, tf.float32) - 127.5) / 128)
    image_batch = image_batch.map(
        lambda image_feature: image_color_distort(image_feature))
    image_batch = image_batch.batch(batch_size)

    # 进行lable部分的操作
    lable_batch = parsed_image_dataset.map(
        lambda image_feature: tf.cast(image_feature['image/label'], tf.float32))
    lable_batch = lable_batch.batch(batch_size)

    roi_batch = parsed_image_dataset.map(
        lambda image_feature: tf.cast(image_feature['image/roi'], tf.float32))
    roi_batch = roi_batch.batch(batch_size)

    landmark_batch = parsed_image_dataset.map(
        lambda image_feature: tf.cast(image_feature['image/landmark'], tf.float32))
    landmark_batch = landmark_batch.batch(batch_size)

    return image_batch, lable_batch, roi_batch, landmark_batch

def read_images_to_tensor(image_list):
    total_image_vec = []
    for image_file in image_list:
        img_vec = cv2.imread(image_file)
        total_image_vec.append(img_vec)
    return tf.data.Dataset.from_tensor_slices(total_image_vec)

if __name__ == '__main__':
    print(tf.executing_eagerly())

    image_batch, lable_batch, roi_batch, landmark_batch = read_tf(
        './DATA/imglists/PNet/train_PNet_landmark.tfrecord_shuffle', 12, 32)
    
    print(list(lable_batch.as_numpy_iterator())[0:2])
