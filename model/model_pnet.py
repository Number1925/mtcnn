import os
from operator import iadd
import tensorflow as tf
import datetime

from tensorflow.python.keras.engine import training
import config.pnet_config_local as config

from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model
from keras.layers.advanced_activations import PReLU
from tensorflow.python.keras.backend import exp
from data_prepare.read_tfrecords import read_tf

MODEL_BASIC_PATH = './save_models/pnet'
batch_size = config.batch_size
num_keep_radio = config.num_keep_radio
radio_cls_loss = config.radio_cls_loss
radio_bbox_loss = config.radio_bbox_loss
radio_landmark_loss = config.radio_landmark_loss
EPOCHS = 30
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

m_cls_loss = tf.keras.metrics.Mean(name='cls_loss')
m_bbox_loss = tf.keras.metrics.Mean(name='bbox_loss')
m_landmark_loss = tf.keras.metrics.Mean(name='landmark_loss')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join(
    config.tensorboard_log_path, current_time, 'train')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


class PNet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = Conv2D(
            10, 3, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))
        self.prelu1 = PReLU(shared_axes=[1, 2])
        self.maxPool1 = MaxPool2D(strides=(2, 2), padding='same')

        self.conv2 = Conv2D(
            16, 3, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))
        self.prelu2 = PReLU(shared_axes=[1, 2])

        self.conv3 = Conv2D(
            32, 3, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))
        self.prelu3 = PReLU(shared_axes=[1, 2])

        self.conv4_1 = Conv2D(2, 1, activation='softmax',
                              kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))
        self.bbox_pred = Conv2D(
            4, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))
        self.landmark_pred = Conv2D(
            10, 1, kernel_regularizer=tf.keras.regularizers.l2(l=0.0005))

    def call(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.maxPool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        class_pred = self.conv4_1(x)
        bbox_pred = self.bbox_pred(x)
        landmark_pred = self.landmark_pred(x)
        return class_pred, bbox_pred, landmark_pred

    def def_loss_func(self, class_pred, class_label, bbox_pred, bbox_label, landmark_pred, landmark_label):
        cls_prob = tf.squeeze(class_pred, [1, 2], name='cls_prob')
        cls_loss = self.cls_ohem(cls_prob, class_label)

        bbox_pred = tf.squeeze(bbox_pred, [1, 2], name='bbox_pred')
        bbox_loss = self.bbox_ohem(bbox_pred, bbox_label, class_label)

        landmark_pred = tf.squeeze(landmark_pred, [1, 2], name="landmark_pred")
        landmark_loss = self.landmark_ohem(
            landmark_pred, landmark_label, class_label)

        accuracy = self.cal_accuracy(cls_prob, class_label)
        return cls_loss, bbox_loss, landmark_loss, accuracy

    def calcu_net_loss(self, cls_loss, bbox_loss, landmark_loss):
        loss = radio_cls_loss*self.filter_nan_tensor(cls_loss)+radio_bbox_loss * \
            self.filter_nan_tensor(bbox_loss)+radio_landmark_loss * \
            self.filter_nan_tensor(landmark_loss)
        return loss

    def filter_nan_tensor(self, t):
        zeros = tf.zeros_like(t)
        return tf.where(tf.math.is_nan(t), zeros, t)

    def cls_ohem(self, cls_prob, label):
        zeros = tf.zeros_like(label)
        # 如果label值大于0保留原值 如果小于0则置为0
        label_filter_invalid = tf.where(tf.less(label, 0), zeros, label)
        num_cls_prob = tf.size(cls_prob)  # 预测向量的元素个数
        cls_prob_reshape = tf.reshape(
            cls_prob, [num_cls_prob, -1])  # 打平成 n*1 的矩阵
        label_int = tf.cast(label_filter_invalid, tf.int32)
        # 获取label的行数 并声明为tensor
        num_row = tf.convert_to_tensor(cls_prob.get_shape()[0], dtype=tf.int32)
        # row = [0,2,4.....]
        row = tf.range(num_row)*2
        # 预测向量为[无人脸的置信分，有人脸置信分] 通过indices来选择label对应的概率
        indices_ = row + label_int
        label_prob = tf.squeeze(
            tf.gather(cls_prob_reshape, indices_))  # 通过gather来获取对应的置信分
        loss = -tf.math.log(label_prob+1e-10)
        zeros = tf.zeros_like(label_prob, dtype=tf.float32)
        ones = tf.ones_like(label_prob, dtype=tf.float32)
        # 标记这个batch中正样本
        valid_inds = tf.where(label < zeros, zeros, ones)
        # 获取所有的正样本
        num_valid = tf.reduce_sum(valid_inds)

        keep_num = tf.cast(num_valid*num_keep_radio, dtype=tf.int32)
        # 过滤landmark的样本
        loss = loss * valid_inds
        loss, _ = tf.nn.top_k(loss, k=keep_num)
        return tf.reduce_mean(loss)

    def bbox_ohem(self, bbox_pred, bbox_target, label):
        zeros_index = tf.zeros_like(label, dtype=tf.float32)
        ones_index = tf.ones_like(label, dtype=tf.float32)
        # 保留带有人脸的样本
        valid_inds = tf.where(tf.equal(tf.abs(label), 1),
                              ones_index, zeros_index)
        # (batch,)
        # 计算两点间的空间距离
        square_error = tf.square(bbox_pred-bbox_target)
        square_error = tf.reduce_sum(square_error, axis=1)
        # 正样本数量
        num_valid = tf.reduce_sum(valid_inds)
        #
        #
        keep_num = tf.cast(num_valid, dtype=tf.int32)
        # 保留全部正样本的loss
        square_error = square_error*valid_inds
        # 保留全部的正样本
        _, k_index = tf.nn.top_k(square_error, k=keep_num)
        square_error = tf.gather(square_error, k_index)

        return tf.reduce_mean(square_error)

    def landmark_ohem(self, landmark_pred, landmark_target, label):
        # 只保留label为-2的样本，计算空间距离作为loss
        ones = tf.ones_like(label, dtype=tf.float32)
        zeros = tf.zeros_like(label, dtype=tf.float32)
        valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
        square_error = tf.square(landmark_pred-landmark_target)
        square_error = tf.reduce_sum(square_error, axis=1)
        num_valid = tf.reduce_sum(valid_inds)
        #keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
        keep_num = tf.cast(num_valid, dtype=tf.int32)
        square_error = square_error*valid_inds
        _, k_index = tf.nn.top_k(square_error, k=keep_num)
        square_error = tf.gather(square_error, k_index)
        return tf.reduce_mean(square_error)

    def cal_accuracy(self, cls_prob, label):
        # get the index of maximum value along axis one from cls_prob
        # 0 for negative 1 for positive
        pred = tf.argmax(cls_prob, axis=1)
        label_int = tf.cast(label, tf.int64)
        # return the index of pos and neg examples
        cond = tf.where(tf.greater_equal(label_int, 0))
        picked = tf.squeeze(cond)
        # gather the label of pos and neg examples
        label_picked = tf.gather(label_int, picked)
        pred_picked = tf.gather(pred, picked)
        # calculate the mean value of a vector contains 1 and 0, 1 for correct classification, 0 for incorrect
        # ACC = (TP+FP)/total population
        accuracy_op = tf.reduce_mean(
            tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
        return accuracy_op


def train_step(net, optimizer, image_list, label_list, bbox_list, landmark_list):
    index = 0
    for image in image_list:
        with tf.GradientTape() as tape:
            class_pred, bbox_pred, landmark_pred = net(image, training=True)
            cls_loss, bbox_loss, landmark_loss, accuracy = net.def_loss_func(
                class_pred, label_list[index], bbox_pred, bbox_list[index], landmark_pred, landmark_list[index])
            loss = net.calcu_net_loss(cls_loss, bbox_loss, landmark_loss)
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))

        m_cls_loss(cls_loss)
        m_bbox_loss(bbox_loss)
        m_landmark_loss(landmark_loss)
        train_loss(loss)
        train_accuracy(accuracy)
        index = index+1


def train():
    net = PNet()

    for epoch in range(EPOCHS):
        image_batch, lable_batch, roi_batch, landmark_batch = read_tf(
            './DATA/imglists/PNet/train_PNet_landmark.tfrecord_shuffle', 12, batch_size)

        image_list = []
        label_list = []
        bbox_list = []
        landmark_list = []
        for image in image_batch:
            image_list.append(image)

        for lable in lable_batch:
            label_list.append(lable)

        for bbox in roi_batch:
            bbox_list.append(bbox)

        for landmark in landmark_batch:
            landmark_list.append(landmark)

        train_loss.reset_states()
        train_accuracy.reset_states()
        train_step(net, optimizer, image_list,
                   label_list, bbox_list, landmark_list)

        with train_summary_writer.as_default():
            tf.summary.scalar('cls_loss', m_cls_loss.result(), step=epoch)
            tf.summary.scalar('bbox_loss', m_bbox_loss.result(), step=epoch)
            tf.summary.scalar(
                'landmark_loss', m_landmark_loss.result(), step=epoch)
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        print(
            f'Epoch {epoch + 1}, '
            f'cls_Loss: {m_cls_loss.result()}, '
            f'bbox_Loss: {m_bbox_loss.result()}, '
            f'landmark_Loss: {m_landmark_loss.result()}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
        )

    print('Stop train and start to save model...')
    model_path = os.path.join(MODEL_BASIC_PATH, "1/")
    keras_model_path = os.path.join(MODEL_BASIC_PATH, "keras1/")
    tf.saved_model.save(net, model_path)
    net.save_weights(keras_model_path)
    print('Model has been saved at %s', model_path)


def predict():
    model_path = os.path.join(MODEL_BASIC_PATH, "1/")
    loaded = tf.saved_model.load(model_path)
    print('Start to load model which at ', model_path)
    print(list(loaded.signatures.keys()))
    model = loaded.signatures["serving_default"]

    image_batch, lable_batch, roi_batch, landmark_batch = read_tf(
        './DATA/imglists/PNet/train_PNet_landmark.tfrecord_shuffle', 12, 1)
    for image in image_batch:
        class_pred = model(image)['output_1']
        print('class_pred:{0}'.format(class_pred))
        break


def predict_by_keras():
    net = PNet()
    model_path = os.path.join(MODEL_BASIC_PATH, "keras1/")
    net.load_weights(model_path)

    image_batch, lable_batch, roi_batch, landmark_batch = read_tf(
        './DATA/imglists/PNet/train_PNet_landmark.tfrecord_shuffle', 12, 1)
    for image in image_batch:
        class_pred, _, _ = net(image, training=False)
        print('class_pred:{0}'.format(class_pred))
        break


if __name__ == '__main__':
    # train()
    # predict()
    predict_by_keras()
