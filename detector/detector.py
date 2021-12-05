import tensorflow as tf
import numpy as np


class Detector:
    def __init__(self, model_path, net=None):
        """[summary]

        Args:
            model ([Model]): [Specific model which you can see in Module named model.]
            data_size ([type]): [description]
            batch_size ([type]): [description]
            model_path ([type]): [description]
        """
        print('Start to load model which at ', model_path)
        if net:
            net.load_weights(model_path)
            self.model = net
        # else:
        #     loaded = tf.saved_model.load(model_path)
        #     print(list(loaded.signatures.keys()))
        #     self.model = loaded.signatures["serving_default"]

    def predict(self, input):
        conv4_1, bbox_pred, landmark_pred = self.model(input)
        cls_pro_test = tf.squeeze(conv4_1, axis=0)
        bbox_pred_test = tf.squeeze(bbox_pred, axis=0)
        landmark_pred_test = tf.squeeze(landmark_pred, axis=0)
        return cls_pro_test.numpy(), bbox_pred_test.numpy()
