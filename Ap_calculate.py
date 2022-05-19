# -*- coding: utf-8 -*-
# @Time : 2022/5/5 10:30
# @Author : Chenglong She
# @File : Ap_calculate.py
# @Software: PyCharm
# 计算AP的方法

import cv2
import os
import numpy as np
from skimage import io
import copy
import time
import matplotlib.pyplot as plt

pre_path = r'E:\CELL\Test\model\val_mask_unet'  # 预测mask的文件夹
gt_path = r'E:\CELL\Test\model\masks'  # ground truth(GT) mask文件夹
img_size = (224, 224)  # 图像的尺寸（只需要长宽）
classes = np.array([0, 1]).astype('uint8')  # 每一类的灰度值表示
files = os.listdir(pre_path)
mAP = []  # 创建mAP变量，该变量实际意义为AP，不是经典的mAP指标，是指将所有测试集计算出来的AP取平均值

# iou阈值的遍历，从iou=0.5到iou=1.0，每次步进0.05
for i_iou in range(0, 11):
    AP = []
    start = time.time()

    # 遍历测试集中所有的图片
    for i, file in enumerate(files):
        iou_threshold = 0.5 + i_iou*0.05  # 每次步进0.05

        # 与计算AP相关变量的初始化
        TP_mask = 0
        FP_mask = 0
        FN_mask = 0

        # 以灰度值的形式读取mask
        img1 = io.imread(os.path.join(pre_path, file), as_gray=True)
        img2 = io.imread(os.path.join(gt_path, file), as_gray=True)
        label_dict_pre = {}
        label_dict_gt = {}

        # 计算每个标签子图和GT图的mask数量，unique_pre代表的是predicted mask的像素值，
        # count_pre代表的是predicted mask中不同像素值的像素个数
        unique_pre, count_pre = np.unique(img1, return_counts=True)
        unique_gt, count_gt = np.unique(img2, return_counts=True)

        data_count_pre = dict(zip(unique_pre, count_pre))
        del data_count_pre[0]  # 将像素值为0的背景删掉
        data_count_gt = dict(zip(unique_gt, count_gt))
        del data_count_gt[0]  # 将像素值为0的背景删掉

        # 将GT与标签中的像素值存储起来做成字典的数据类型，label_dict_pre变量为{'0':0, '1':1, ..., 'N':N}
        for index in data_count_pre.keys():
            label_dict_pre['{}'.format(index)] = index
        for index in data_count_gt.keys():
            label_dict_gt['{}'.format(index)] = index
        label_key_pre = list(label_dict_pre.keys())
        label_value_pre = list(label_dict_pre.values())
        label_key_gt = list(label_dict_gt.keys())
        label_value_gt = list(label_dict_gt.values())
        
        # 遍历一张测试集图片中的所有mask，每一个mask的label都不一样，按自然数顺序排列，该层循环是计算TP，FP。FP+TP等于全部的predicted mask
        # TP代表IoU大于阈值IoU(0.50, 0.55, ..., 1.0, 每次循环增加0.05的步距)的predicted mask数, FP代表IoU小于等于阈值的predicted mask数
        for value_index, value in enumerate(label_value_pre):  # 循环predict中的每一个mask
            non_label_index_pre = np.argwhere(img1 != value)  # 获取mask矩阵中不等于value(value即为label)的坐标
            temp_mask_pre = copy.deepcopy(img1)
            temp_mask_pre[non_label_index_pre[:, 0], non_label_index_pre[:, 1]] = 0  # 将遍历到的当前mask之外的区域全部填0变成背景
            iou_max = 0  # 初始化iou最大值

            # 循环GT中的每一个mask, 将标签图中的一个mask与GT中的每个mask作IoU计算，将IoU最大的一对mask进行匹配
            for v_index, v in enumerate(label_value_gt):  
                non_label_index_gt = np.argwhere(img2 != v)  # 获取mask矩阵中不等于v（v即为label）的坐标
                temp_mask_gt = copy.deepcopy(img2)
                temp_mask_gt[non_label_index_gt[:, 0], non_label_index_gt[:, 1]] = 0

                # 将两个mask的像素值都置1，方便后面算IoU
                temp_mask_pre = np.int16(temp_mask_pre > 0)
                temp_mask_gt = np.int16(temp_mask_gt > 0)

                # 初始化TP_mat 这个变量和前面的TP_mask不一样，主要计算的是两个mask的重合像素个数
                TP_mat = temp_mask_pre * (temp_mask_pre == temp_mask_gt)

                # 如果两个mask有交叠才计算，减少计算成本
                if TP_mat.max() != 0:
                    FP_mat = temp_mask_pre * (temp_mask_pre != temp_mask_gt)
                    FN_mat = temp_mask_gt * (temp_mask_pre != temp_mask_gt)
                    (TP, _) = np.histogram(TP_mat, bins=2, range=(0, 1))  # 这里得到的是交叠区域的像素点个数
                    (FP, _) = np.histogram(FP_mat, bins=2, range=(0, 1))  # 这里得到的是predicted mask减去交叠区域的像素点个数
                    (FN, _) = np.histogram(FN_mat, bins=2, range=(0, 1))  # 这里得到的是GT mask减去交叠区域的像素点个数
                    iou_tmp = (TP / (TP + FP + FN))[0:2][1]  # 计算得到当前两个mask的IoU值
                    if iou_tmp > iou_max:  # 与前一个计算的Iou进行比较
                        iou_max = iou_tmp
            if iou_max > iou_threshold:
                TP_mask += 1
        FP_mask = len(label_key_pre)-TP_mask

        # 循环GT中的每一个mask，该层循环是计算FN，若GT中当前的mask与标签图中的任何一个mask都没有交集，则判断该GT中的mask为FN
        for v_index, v in enumerate(label_value_gt):  
            non_label_index_gt = np.argwhere(img2 != v)  # 获取mask矩阵中不等于v（v即为label）的坐标
            temp_mask_gt = copy.deepcopy(img2)
            temp_mask_gt[non_label_index_gt[:, 0], non_label_index_gt[:, 1]] = 0
            flag = 0
            for value_index, value in enumerate(label_value_pre):  # 循环pre中的每一个mask
                non_label_index_pre = np.argwhere(img1 != value)  # 获取mask矩阵中不等于v（v即为label）的坐标
                temp_mask_pre = copy.deepcopy(img1)
                temp_mask_pre[non_label_index_pre[:, 0], non_label_index_pre[:, 1]] = 0

                temp_mask_pre = np.int16(temp_mask_pre > 0)
                temp_mask_gt = np.int16(temp_mask_gt > 0)
                TP_mat = temp_mask_pre * (temp_mask_pre == temp_mask_gt)
                if TP_mat.max() != 0:
                    flag = 1
            if flag == 0:
                FN_mask += 1

        # AP_tmp变量代表该图的AP
        AP_tmp = TP_mask/(TP_mask + FP_mask + FN_mask)

        # AP为列表变量，存储该IoU阈值下每张测试集图片的AP
        AP.append(AP_tmp)
        print("index in one iter:", i)
    print("consuming TIME:", time.time() - start)

    # 将该IoU阈值下，测试集中所有的图片的AP取均值
    AP = np.mean(AP)
    print(
            "iou_threshold:{0},AP={1}\n".format(iou_threshold, AP)
          )

    # mAP为列表变量，存储每个IoU阈值下、测试集取均值的AP
    mAP.append(AP)

# # 手动记录了mAP变量最后的结果
# AP_cellpose = [0.81, 0.80, 0.75, 0.71, 0.65, 0.57, 0.46, 0.32, 0.14, 0.01, 0]
# AP_unet = [0.067, 0.050, 0.034, 0.023, 0.011, 0.0056, 0.0025, 0.0011, 0.00078, 0, 0]
# IoU_list = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]

# 改名字的功能
# # cellpose_predicted_path = r'E:\CELL\Test\model\val_cellpose'
# unet_predicted_path = r'E:\CELL\Test\model\masks_unet'
# #
# # cellpose_directory = list(os.walk(cellpose_predicted_path))
# # cellpose_root = cellpose_directory[0][0]
# # cellpose_files = cellpose_directory[0][2]
# #
# unet_directory = list(os.walk(unet_predicted_path))
# unet_root = unet_directory[0][0]
# unet_files = unet_directory[0][2]
#
# # cellpose_mask = cv2.imread(r'E:\CELL\Test\model\val\0001_0001_3_4_cp_masks.png', cv2.IMREAD_ANYDEPTH)
# # unt_mask = cv2.imread(r'E:\CELL\Test\model\val_unet\0001_0001_3_4_cp_masks.png', cv2.IMREAD_ANYDEPTH)
# # GT = cv2.imread(r'E:\CELL\Test\masks\0001_0001_3_4_masks.png', cv2.IMREAD_ANYDEPTH)
# # pass
# for i in unet_files:
#     if 'masks' in i:
#         oldname = unet_predicted_path + '\\' + i
#         a = i.replace('_cp_', '_')
#         newname = unet_predicted_path + '\\' + a
#         os.rename(oldname, newname)
#         pass