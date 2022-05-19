# -*- coding: utf-8 -*-
"""
将数据集划分为训练集，验证集，测试集
"""

import os
import random
import shutil
# 创建保存图像的文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
random.seed(1) # 随机种子

# 1.确定原图像数据集路径
dataset_dir = "/jdfssz1/ST_SUPERCELLS/P21Z10200N0171/USER/shechenglong/mask_rcnn/tool/data"  ##原始数据集路径
# 2.确定数据集划分后保存的路径
split_dir = "/jdfssz1/ST_SUPERCELLS/P21Z10200N0171/USER/shechenglong/mask_rcnn/tool"  ##划分后保存路径
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "val")
test_dir = os.path.join(split_dir, "test")
# 3.确定将数据集划分为训练集，验证集，测试集的比例
train_pct = 0.8
valid_pct = 0.2
test_pct = 0
# 4.划分

imgs = os.listdir(dataset_dir)  # 展示目标文件夹下所有的文件名
imgs = list(filter(lambda x: x.endswith('.png'), imgs))  # 取到所有以.png结尾的文件，如果改了图片格式，这里需要修改
random.shuffle(imgs)  # 乱序图片路径
img_count = len(imgs)  # 计算图片数量
train_point = int(img_count * train_pct)  # 0:train_pct
valid_point = int(img_count * (train_pct + valid_pct))  # train_pct:valid_pct

for i in range(img_count):
    if i < train_point:  # 保存0-train_point的图片到训练集
        out_dir = os.path.join(train_dir)
    elif i < valid_point:  # 保存train_point-valid_point的图片到验证集
        out_dir = os.path.join(valid_dir)
    else:  # 保存valid_point-结束的图片到测试集
        out_dir = os.path.join(test_dir)
    makedir(out_dir)  # 创建文件夹
    target_path = os.path.join(out_dir, imgs[i])  # 指定目标保存路径
    js = imgs[i].split('.')[0]+'.json'
    target_path_json = os.path.join(out_dir, js)  # json目标路径
    src_path = os.path.join(dataset_dir, imgs[i])  # 指定目标原图像路径
    src_path_json = os.path.join(dataset_dir, js)
    shutil.copy(src_path, target_path)  # 复制图片
    shutil.copy(src_path_json, target_path_json)

print('train:{}, valid:{}, test:{}'.format(train_point, valid_point-train_point,
                                                     img_count-valid_point))
