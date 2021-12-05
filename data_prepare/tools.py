# coding:utf-8
import os
import cv2
import numpy as np

# 工具函数汇总


def mkdir_wiz_parent(path):
    """
        创建文件夹,当上一级文件不存在时一并创建
    """
    path_list = path.split('/')
    for i in range(len(path_list)):
        i = i+1
        if i > 1:
            part_of_path = '/'.join(path_list[0:i])
            import os
            if not os.path.exists(part_of_path):
                os.mkdir(part_of_path)


def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes
    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        predicted boxes
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes
    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def image_xywh_xyxy(box):
    """
        转换box的结构从 [x1, y1, w, h] 转换成 [x_left, y_top, x_right, y_bottom]
    """
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    return [x, y, str(int(x) + int(w)), str(int(y) + int(h))]


def get_data_from_txt(txt, data_path, with_landmark=True):
    with open(os.path.join(data_path, txt), 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(data_path, components[0]).replace(
            '\\', '/')  # file path

        # bounding box, (x1, y1, x2, y2)
        #bbox = (components[1], components[2], components[3], components[4])
        bbox = (components[1], components[3], components[2], components[4])
        bbox = [float(_) for _ in bbox]
        bbox = list(map(int, bbox))
        # landmark
        if not with_landmark:
            result.append((img_path, BBox(bbox)))
            continue
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        # normalize
        '''
        for index, one in enumerate(landmark):
            rv = ((one[0]-bbox[0])/(bbox[2]-bbox[0]), (one[1]-bbox[1])/(bbox[3]-bbox[1]))
            landmark[index] = rv
        '''
        result.append((img_path, BBox(bbox), landmark))
    return result


class BBox(object):
    """
        Bounding Box of face
    """

    def __init__(self, bbox):
        self.left = bbox[0]
        self.top = bbox[1]
        self.right = bbox[2]
        self.bottom = bbox[3]

        self.x = bbox[0]
        self.y = bbox[1]
        self.w = bbox[2] - bbox[0]
        self.h = bbox[3] - bbox[1]

    def expand(self, scale=0.05):
        bbox = [self.left, self.right, self.top, self.bottom]
        bbox[0] -= int(self.w * scale)
        bbox[1] += int(self.w * scale)
        bbox[2] -= int(self.h * scale)
        bbox[3] += int(self.h * scale)
        return BBox(bbox)
    # offset

    def project(self, point):
        x = (point[0]-self.x) / self.w
        y = (point[1]-self.y) / self.h
        return np.asarray([x, y])
    # absolute position(image (left,top))

    def reproject(self, point):
        x = self.x + self.w*point[0]
        y = self.y + self.h*point[1]
        return np.asarray([x, y])
    #landmark: 5*2

    def reprojectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.reproject(landmark[i])
        return p
    # change to offset according to bbox

    def projectLandmark(self, landmark):
        p = np.zeros((len(landmark), 2))
        for i in range(len(landmark)):
            p[i] = self.project(landmark[i])
        return p
    #f_bbox = bbox.subBBox(-0.05, 1.05, -0.05, 1.05)
    # self.w bounding-box width
    # self.h bounding-box height

    def subBBox(self, leftR, rightR, topR, bottomR):
        leftDelta = self.w * leftR
        rightDelta = self.w * rightR
        topDelta = self.h * topR
        bottomDelta = self.h * bottomR
        left = self.left + leftDelta
        right = self.left + rightDelta
        top = self.top + topDelta
        bottom = self.top + bottomDelta
        return BBox([left, right, top, bottom])

#rotate(img, f_bbox,bbox.reprojectLandmark(landmarkGt), 5)
# img: the whole image
# BBox:object
# landmark:
# alpha:angle


def rotate(img, bbox, landmark, alpha):
    """
        given a face with bbox and landmark, rotate with alpha
        and return rotated face with bbox, landmark (absolute position)
    """
    center = ((bbox.left+bbox.right)/2, (bbox.top+bbox.bottom)/2)
    rot_mat = cv2.getRotationMatrix2D(center, alpha, 1)
    # whole image rotate
    # pay attention: 3rd param(col*row)
    img_rotated_by_alpha = cv2.warpAffine(
        img, rot_mat, (img.shape[1], img.shape[0]))
    landmark_ = np.asarray([(rot_mat[0][0]*x+rot_mat[0][1]*y+rot_mat[0][2],
                             rot_mat[1][0]*x+rot_mat[1][1]*y+rot_mat[1][2]) for (x, y) in landmark])
    # crop face
    face = img_rotated_by_alpha[bbox.top:bbox.bottom+1, bbox.left:bbox.right+1]
    return (face, landmark_)


def flip(face, landmark):
    """
        flip face
    """
    face_flipped_by_x = cv2.flip(face, 1)
    # mirror
    landmark_ = np.asarray([(1-x, y) for (x, y) in landmark])
    landmark_[[0, 1]] = landmark_[[1, 0]]  # left eye<->right eye
    landmark_[[3, 4]] = landmark_[[4, 3]]  # left mouth<->right mouth
    return (face_flipped_by_x, landmark_)


def read_annotation(base_dir, label_path):
    """
    read label file
    :param dir: path
    :return:
    """
    data = dict()
    images = []
    bboxes = []
    labelfile = open(label_path, 'r')
    while True:
        # 图片地址
        imagepath = labelfile.readline().strip('\n')
        if not imagepath:
            break
        imagepath = base_dir + '/WIDER_train/images/' + imagepath
        
        # 人脸数量
        nums = labelfile.readline().strip('\n')
        if int(nums) == 0:
            bb_info = labelfile.readline().strip('\n').split(' ')
            continue
        else:
            images.append(imagepath)
        one_image_bboxes = []
        for i in range(int(nums)):
            # box位置
            bb_info = labelfile.readline().strip('\n').split(' ')
            # only need x, y, w, h
            face_box = [float(bb_info[i]) for i in range(4)]
            xmin = face_box[0]
            ymin = face_box[1]
            xmax = xmin + face_box[2]
            ymax = ymin + face_box[3]
            one_image_bboxes.append([xmin, ymin, xmax, ymax])
        bboxes.append(one_image_bboxes)

    data['images'] = images  # all images
    data['bboxes'] = bboxes  # all image bboxes
    return data
