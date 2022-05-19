# -*- coding: utf-8 -*-
# @Time : 2022/5/5 10:30
# @Author : Chenglong She
# @File : json_modified.py
# @Software: PyCharm
# 将outlines2json.py文件生成的mask为全“1”label的json文件，改成mask以自然数序数增长的json，主要是修改label的功能
import os
import cv2
import json

current_path = os.path.dirname(os.path.abspath(__file__))#获得当前脚本所在的目录路径
img_dir = os.path.join(current_path, 'img/')

#获取json文件的文件名
json_files = []
all_files = os.listdir(img_dir)
for file in all_files:
    if "json" in file:
        json_files.append(file)

#遍历json文件并修改label，将label由若干个“1”递归成“1”“2”“3”...
for id in json_files:
    json_path = img_dir + id
    save_path = json_path #直接替换掉原来的json文件
    with open(json_path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    for index in range(len(load_dict['shapes'])):
        load_dict['shapes'][index]['label'] = str(index+1)# 该句为主要替换动作，若想改成全“1”label，则将str(1)赋值给等号前边的变量即可
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(load_dict, f, ensure_ascii=False)