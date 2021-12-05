from tensorflow.python.framework.ops import Tensor
import data_prepare.read_tfrecords as read_tools
import tensorflow as tf
import numpy as np
import time
import cv2

from model.model_pnet import predict

def py_nms(dets, thresh, mode="Union"):
        """
        greedily select boxes with high confidence
        keep boxes overlap <= thresh
        rule out overlap > thresh
        :param dets: [[x1, y1, x2, y2 score]]
        :param thresh: retain overlap <= thresh
        :return: indexes to keep
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            if mode == "Union":
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == "Minimum":
                ovr = inter / np.minimum(areas[i], areas[order[1:]])
            #keep
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep
    
class MtcnnDetector(object):

    def __init__(self,
                 detectors,
                 min_face_size=20,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79,
                 slide_window=False):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
        self.slide_window = slide_window
        
    def detect_face(self, image_file_list):
        all_boxes = []  # save each image's bboxes
        landmarks = []
        batch_idx = 0

        sum_time = 0
        t1_sum = 0
        t2_sum = 0
        t3_sum = 0
        num_of_img = len(image_file_list)
        empty_array = np.array([])
        s_time = time.time()
        
        # Need to create a struct that can iterate specific number of rows data
        # Otherwise the memeroy is not enough
        for image_t in image_file_list:
            batch_idx += 1
            if batch_idx % 100 == 0:
                c_time = (time.time() - s_time )/100
                print("%d out of %d images done" % (batch_idx ,num_of_img))
                print('%f seconds for each image' % c_time)
                s_time = time.time()

            if self.pnet_detector:
                st = time.time()
                boxes, boxes_c, landmark = self.detect_pnet(image_t)

                t1 = time.time() - st
                sum_time += t1
                t1_sum += t1
                if boxes_c is None:
                    print("boxes_c is None...")
                    all_boxes.append(empty_array)
                    # pay attention
                    landmarks.append(empty_array)

                    continue
                #print(all_boxes)

            # rnet

            # if self.rnet_detector:
            #     t = time.time()
            #     # ignore landmark
            #     boxes, boxes_c, landmark = self.detect_rnet(im, boxes_c)
            #     t2 = time.time() - t
            #     sum_time += t2
            #     t2_sum += t2
            #     if boxes_c is None:
            #         all_boxes.append(empty_array)
            #         landmarks.append(empty_array)

            #         continue
            # # onet

            # if self.onet_detector:
            #     t = time.time()
            #     boxes, boxes_c, landmark = self.detect_onet(im, boxes_c)
            #     t3 = time.time() - t
            #     sum_time += t3
            #     t3_sum += t3
            #     if boxes_c is None:
            #         all_boxes.append(empty_array)
            #         landmarks.append(empty_array)

            #         continue

            all_boxes.append(boxes_c)
            landmark = [1]
            landmarks.append(landmark)
        print('num of images', num_of_img)
        print("time cost in average" +
            '{:.3f}'.format(sum_time/num_of_img) +
            '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1_sum/num_of_img, t2_sum/num_of_img,t3_sum/num_of_img))


        # num_of_data*9,num_of_data*10
        print('boxes length:',len(all_boxes))
        return all_boxes, landmarks
    
    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: batch image tensor

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """
        net_size = 12

        # find initial scale
        current_scale = float(net_size) / self.min_face_size
        # risize image using current_scale
        # why need to resize image to scale
        im_resized = self.processed_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        # It will filter image which high or width is less than mini_face_size
        while min(current_height, current_width) > net_size:
            # return the result predicted by pnet
            # cls_cls_map : H*w*2
            # reg: H*w*4
            # class_prob andd bbox_pred
            im_resized_dataset = tf.data.Dataset.from_tensor_slices([im_resized])
            for im_resized in im_resized_dataset:
                im_resized = tf.reshape(im_resized,[1,current_height,current_width,3])
                im_resized = tf.cast(im_resized,dtype=tf.float32)
                cls_cls_map, reg = self.pnet_detector.predict(im_resized)
                # boxes: num*9(x1,y1,x2,y2,score,x1_offset,y1_offset,x2_offset,y2_offset)
                boxes = self.generate_bbox(cls_cls_map[:, :, 1], reg, current_scale, self.thresh[0])
                # scale_factor is 0.79 in default
                current_scale *= self.scale_factor
                im_resized = self.processed_image(im, current_scale)
                current_height, current_width, _ = im_resized.shape

                if boxes.size == 0:
                    continue
                # get the index from non-maximum s
                keep = py_nms(boxes[:, :5], 0.5, 'Union')
                boxes = boxes[keep]
                all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None, None

        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = py_nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]
        boxes = all_boxes[:, :5]

        bbw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bbh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        # refine the boxes
        boxes_c = np.vstack([all_boxes[:, 0] + all_boxes[:, 5] * bbw,
                             all_boxes[:, 1] + all_boxes[:, 6] * bbh,
                             all_boxes[:, 2] + all_boxes[:, 7] * bbw,
                             all_boxes[:, 3] + all_boxes[:, 8] * bbh,
                             all_boxes[:, 4]])
        boxes_c = boxes_c.T

        return boxes, boxes_c, None
    
    def processed_image(self, img, scale):
        '''
        rescale/resize the image according to the scale
        :param img: image
        :param scale:
        :return: resized image
        '''
        height, width, _ = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
        img_resized = (img_resized - 127.5) / 128
        return img_resized

    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it
        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox
        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list
    
    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            generate bbox from feature cls_map according to the threshold
        Parameters:
        ----------
            cls_map: numpy array , n x m 
                detect score for each position
            reg: numpy array , n x m x 4
                bbox
            scale: float number
                scale of this detection
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array
        """
        stride = 2
        # stride = 4
        cellsize = 12
        # cellsize = 25

        # index of class_prob larger than threshold
        t_index = np.where(cls_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        dx1, dy1, dx2, dy2 = [reg[t_index[0], t_index[1], i] for i in range(4)]

        reg = np.array([dx1, dy1, dx2, dy2])
        score = cls_map[t_index[0], t_index[1]]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                                 np.round((stride * t_index[0]) / scale),
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 score,
                                 reg])

        return boundingbox.T
    
    