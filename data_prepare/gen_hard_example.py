import os
import cv2

import tensorflow as tf
import data_prepare.tools as tools
import argparse
import model.model_pnet as pnet
from detector.detector import Detector
from detector.mtcnn_detector import MtcnnDetector
from data_prepare.sample_reader import SampleReader


def t_net(prefix, epoch,
          batch_size, test_mode="PNet",
          thresh=[0.6, 0.6, 0.7], min_face_size=25,
          stride=2, slide_window=False, shuffle=False, vis=False):
    """[summary]

    Args:
        prefix ([string]): [model param's file]
        epoch ([int]): [final epoches]
        batch_size ([int]): [test batch_size]
        test_mode (str, optional): [test which model]. Defaults to "PNet".
        thresh (list, optional): [cls threshold]. Defaults to [0.6, 0.6, 0.7].
        min_face_size (int, optional): [min_face]. Defaults to 25.
        stride (int, optional): [stride]. Defaults to 2.
        slide_window (bool, optional): [description]. Defaults to False.
        shuffle (bool, optional): [description]. Defaults to False.
        vis (bool, optional): [description]. Defaults to False.
    """
    detectors = [None, None, None]

    # 加在Pnet模型
    # if slide_window:
    PNet = Detector(os.path.join(
        pnet.MODEL_BASIC_PATH, 'keras1/'), pnet.PNet())
    # else:
    #     PNet = FcnDetector(P_Net, model_path[0])
    # 保存模型模型
    detectors[0] = PNet

    basedir = '/home/wangtao/WorkSpace/DataSet/MtcnnData'
    # 标签文件
    filename = '/home/wangtao/WorkSpace/DataSet/MtcnnData/wider_face_split/wider_face_train_bbx_gt.txt'
    # 读取样本文件,转化为dict结构 包含 图片文件地址 坐标信息
    data = tools.read_annotation(basedir, filename)
    mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                   stride=stride, threshold=thresh, slide_window=slide_window)

    sample_reader = SampleReader(data['images'])
    sample_reader.set_batch(1)
    sample_reader.register_map(lambda data: [cv2.imread(x) for x in data])

    for image_data_list in sample_reader:
        detections, _ = mtcnn_detector.detect_face(image_data_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Test mtcnn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test_mode', dest='test_mode', help='test net type, can be pnet, rnet or onet',
                        default='RNet', type=str)
    parser.add_argument('--prefix', dest='prefix', help='prefix of model name', nargs="+",
                        default=['../data/MTCNN_model/PNet_No_Landmark/PNet',
                                 '../data/MTCNN_model/RNet_No_Landmark/RNet', '../data/MTCNN_model/ONet_No_Landmark/ONet'],
                        type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch number of model to load', nargs="+",
                        default=[18, 14, 16], type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='list of batch size used in prediction', nargs="+",
                        default=[2048, 256, 16], type=int)
    parser.add_argument('--thresh', dest='thresh', help='list of thresh for pnet, rnet, onet', nargs="+",
                        default=[0.3, 0.1, 0.7], type=float)
    parser.add_argument('--min_face', dest='min_face', help='minimum face size for detection',
                        default=20, type=int)
    parser.add_argument('--stride', dest='stride', help='stride of sliding window',
                        default=2, type=int)
    parser.add_argument('--sw', dest='slide_window',
                        help='use sliding window in pnet', action='store_true')
    #parser.add_argument('--gpu', dest='gpu_id', help='GPU device to train with',default=0, type=int)
    parser.add_argument('--shuffle', dest='shuffle',
                        help='shuffle data on visualization', action='store_true')
    parser.add_argument('--vis', dest='vis',
                        help='turn on visualization', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    image_size = 24
    base_dir = './DATA/WIDER_train'
    data_dir = './DATA/no_LM%s' % str(image_size)
    neg_dir = os.path.join(data_dir, 'negative')
    pos_dir = os.path.join(data_dir, 'positive')
    part_dir = os.path.join(data_dir, 'part')

    tools.mkdir_wiz_parent(neg_dir)
    tools.mkdir_wiz_parent(pos_dir)
    tools.mkdir_wiz_parent(part_dir)
    args = parse_args()
    print(args)
    t_net(args.prefix,
          args.epoch,
          args.batch_size,
          args.test_mode,
          args.thresh,
          args.min_face,
          args.stride,
          args.slide_window,
          args.shuffle,
          vis=False)
