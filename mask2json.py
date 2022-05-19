# -*- coding: utf-8 -*-
# @Time : 2022/5/5 10:30
# @Author : Chenglong She
# @File : mask2json.py
# @Software: PyCharm
# 用含有若干masks的标签图转成json文件
import base64
import copy
import cv2
import glob
import json
import sys
import time
import cv2 as cv
import os
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import io
import collections

def img_tobyte(self):
    # 类型转换 重要代码
    # img_pil = Image.fromarray(roi)
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    self.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string


class Mask2PolygonsAndSaveLabelMeJson():
    def test_mask_2_labelme_json_by_split_mask(self, mask, img_path, label_dict, save_path):
        """按类拆分mask, 通过获取mask的Contours将标签保存为labelMme所需的json标签"""
        save_label_dict = {'version': '4.6.0',  # 核对自己的labelMe版本
                           'flags': {},
                           'shapes': [],  # [shape_dict]
                           'imagePath': '',
                           'imageData': '',  # 用base64库将cv2读入的npy矩阵转成PIL图像字节数据存入json
                           'imageHeight': self.size[1],
                           'imageWidth': self.size[0]}

        lable_k = list(label_dict.keys())
        label_v = list(label_dict.values())

        for v_i, v in enumerate(label_v):
            list1 = []
            # 按类(或者说label/像素值)拆分mask
            non_label_index = np.argwhere(mask != v) #获取mask矩阵中不等于v（v即为label）的坐标
            temp_mask = copy.deepcopy(mask)
            temp_mask[non_label_index[:, 0], non_label_index[:, 1]] = 0 #将其中一个label的mask保留下来，其余全部填0变黑
            # findContours可以将图中剩下的mask的轮廓像素的坐标保存下来，只不过数量会很大，导致labelme打开时很卡，有需求后续可以筛选抽样点
            contour_tmp, _ = cv.findContours(temp_mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            # 将拥有同一个label的所有masks分别计算面积是否大于3000，大于3000的保留；由于cellpose数据必须是一个mask一个label所以无需过滤
            # contour_list = list(filter(lambda x: cv.contourArea(x) > 3000, contour_tmp))
            # 画图展示
            label_show = lable_k[v_i]
            try:
                for i in range(len(contour_tmp[0])):
                    list1.append(contour_tmp[0][i][0].tolist())
                seg_info = {'points': list1, 'group_id': None, "label": label_show,
                            "shape_type": "polygon", "flags": {}}
                save_label_dict['shapes'].append(seg_info)
            except:
                print(img_path)
        save_label_dict['imagePath'] = img_path.split('\\')[-1]
        save_label_dict['imageData'] = img_tobyte(self)
        with open(save_path, 'w') as output_json_file:
            json.dump(save_label_dict, output_json_file)

IMAGE_DIR = '/jdfssz1/ST_SUPERCELLS/P21Z10200N0171/USER/shechenglong/mask_rcnn/tool/data'
directory = list(os.walk(IMAGE_DIR))
root = directory[0][0]
files = directory[0][2]
for id in files:
    if 'masks' not in id:
        img_path = IMAGE_DIR+'//'+id
        img = Image.open(img_path)
        mask_path = img_path.split('.')[0]+'_masks.png'
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        save_path = img_path.split('.')[0]+'.json'
        label_dict = {}
        unique, count = np.unique(mask, return_counts=True)
        data_count = dict(zip(unique, count))
        del data_count[0]
        for index in data_count.keys():
            label_dict['{}'.format(index)] = index
        a = Mask2PolygonsAndSaveLabelMeJson.test_mask_2_labelme_json_by_split_mask(img, mask, img_path, label_dict, save_path)
