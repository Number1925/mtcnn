# coding=utf-8
import os
import cv2
import numpy as np
import numpy.random as npr
import tools

# 标签文件地址
anno_file = "/home/wangtao/WorkSpace/DataSet/MtcnnData/wider_face_split/wider_face_train_bbx_gt.txt"
# 图片文件夹地址
im_dir = "/home/wangtao/WorkSpace/DataSet/MtcnnData/WIDER_train/images"
# positive 样本存储路径
pos_save_dir = "./DATA/12/positive"
# part 样本存储路径
part_save_dir = "./DATA/12/part"
# negative 样本存储路径
neg_save_dir = './DATA/12/negative'
# 数据存储路径
save_dir = "./DATA/12"

tools.mkdir_wiz_parent(save_dir)
tools.mkdir_wiz_parent(pos_save_dir)
tools.mkdir_wiz_parent(part_save_dir)
tools.mkdir_wiz_parent(neg_save_dir)
f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')

with open(anno_file, 'r') as f:
    annotations = f.readlines()

annotation_list = []

cur_sample_content = []
cur_sample_index = 0
box_line_num = 0
box_num_flag = False
no_target_flag = False
for i in range(len(annotations)):
    if box_line_num == 0 and not box_num_flag:
        cur_sample_content = []
        cur_sample_content.append(annotations[i].strip())
        box_num_flag = True
        continue
    if box_line_num > 0 and box_num_flag:
        # 训练集中的box [x1, y1, w, h]
        box_ori = annotations[i].strip().split(' ')[0:4]
        # 转换box类型为 [x_left, y_top, x_right, y_bottom]
        cur_sample_content.extend(tools.image_xywh_xyxy(box_ori))
        box_line_num = box_line_num - 1
        if box_line_num == 0:
            box_num_flag = False
            if not no_target_flag:
                no_target_flag = False
                annotation_list.append(cur_sample_content)
        continue
    if box_num_flag:
        box_line_num = int(annotations[i].strip())
        if box_line_num == 0:
            no_target_flag = True
            box_line_num = 1

print(annotation_list[0:3])
annotations = annotation_list

num = len(annotations)
print("%d pics in total" % num)
p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # don't care
idx = 0
box_idx = 0
for annotation in annotations:
    # 图片地址
    im_path = annotation[0]
    # 转换图片box类型 to float
    bbox = list(map(float, annotation[1:]))
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    # 加载图片
    img = cv2.imread(os.path.join(im_dir, im_path))
    idx += 1

    height, width, channel = img.shape

    neg_num = 0
    # 1---->50
    # 保留随机剪裁,直至负样本数量到达50个
    while neg_num < 50:
        # 负样本的图片大小区间在 [12,min(width, height) / 2] 之间
        size = npr.randint(12, min(width, height) / 2)
        # 左上角锚点
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        # 随机剪裁的box 左上nx,ny
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # 计算剪裁的框和label中的box的交并比
        Iou = tools.IoU(crop_box, boxes)

        # 剪裁图片
        cropped_im = img[ny: ny + size, nx: nx + size, :]
        # 将剪裁后的图片resize至12*12
        resized_im = cv2.resize(cropped_im, (12, 12),
                                interpolation=cv2.INTER_LINEAR)

        # 剪裁部分和label中的box最大的交并比都小于0.3的情况,则认为为负样本
        if np.max(Iou) < 0.3:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            annotation_content = save_file + ' 0\n'
            f2.write(annotation_content)
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    # 遍历所有box
    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        # 忽略小的box和左上角只有残缺的图片
        if max(w, h) < 20 or x1 < 0 or y1 < 0:
            continue

        # 裁剪出5张box附近,并且iou值小于0.5的图片
        for i in range(5):
            # 同上所述一样,截取大小为[12, min(width, height) / 2]
            size = npr.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            # max can make sure if the delta is a negative number , x1+delta_x >0
            # parameter high of randint make sure there will be intersection between bbox and cropped_box
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            # max here not really necessary
            # 找到裁剪的左上点
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            # 如果裁剪的右下点超出图片范围则放弃
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = tools.IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            # 把裁剪后的图片resize到 12*12
            resized_im = cv2.resize(
                cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                annotation_content = save_file + ' 0\n'
                f2.write(annotation_content)
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # 生成正样本
        for i in range(20):
            # 随机剪裁,剪裁的大小范围为[min(box_w,box_h)*0.8,max(box_w,box_h)*1.25]
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # 跳过宽小于5的box
            if w < 5:
                print(w)
                continue
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)

            # show this way: nx1 = max(x1+w/2-size/2+delta_x)
            # x1+ w/2 is the central point, then add offset , then deduct size/2
            # deduct size/2 to make sure that the right bottom corner will be out of
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            # show this way: ny1 = max(y1+h/2-size/2+delta_y)
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            # yu gt de offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            # crop
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            # resize
            resized_im = cv2.resize(
                cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            iou = tools.IoU(crop_box, box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' %
                         (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' %
                         (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" %
                  (idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
