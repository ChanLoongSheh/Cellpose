# -*- coding: utf-8 -*-
# @Time : 2022/5/5 10:30
# @Author : Chenglong She
# @File : outlines2json.py
# @Software: PyCharm
# 将cellpose的预测的xxxx_cp_outlines.txt转化为.json文件，label为全“1”

import os
import io
import json
from PIL import Image
import base64

def img_tobyte(img_pil):
    # 类型转换 重要代码
    # img_pil = Image.fromarray(roi)
    ENCODING = 'utf-8'
    img_byte = io.BytesIO()
    img_pil.save(img_byte, format='PNG')
    binary_str2 = img_byte.getvalue()
    imageData = base64.b64encode(binary_str2)
    base64_string = imageData.decode(ENCODING)
    return base64_string

current_path = os.path.dirname(os.path.abspath(__file__))#获得当前脚本所在的目录路径
IMAGE_DIR = os.path.join(current_path, 'img/') #为了让PIL读取图片数据然后转成字节数据存入json

# 判断是否存在img文件夹如果不存在则创建为文件夹
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

annotation_files = []
all_files = os.listdir(IMAGE_DIR)
for file in all_files:
    if "outlines" in file:
        annotation_files.append(file)

for annotation_filename in annotation_files: #遍历img文件夹下的txt文件
    coco_output = {                         #这个是json文件中每个mask记录的格式
        "version": "4.6.0",
        "flags": {},
        "fillColor": [255, 0, 0, 128],
        "lineColor": [0, 255, 0, 128],
        "imagePath": {},
        "shapes": [],
        "imageData": {}}

    print(annotation_filename)
    name = annotation_filename.split('_')[0] + "_" + annotation_filename.split('_')[1] #这个是找出切片的编号名称
    image_name = name + '.png'
    coco_output["imagePath"] = image_name

    image = Image.open(IMAGE_DIR + '/' + image_name)
    imageData = img_tobyte(image) #将图像作为字节对象返回，json文件的格式需要这个字节对象部分
    coco_output["imageData"] = imageData

    segmentation = []
    f = open(IMAGE_DIR+name+'_cp_outlines.txt', encoding="utf-8")
    lines = f.readlines() #将单个outlines文件每一行作为一个元素编入lines（数据类型为list），每一行即为一个mask的轮廓坐标
    for l in lines:
        txt_data = l.split(',') #每个mask的坐标集合，该变量的shape()属性除以2即为该围成该mask的轮廓的点的数量，（x1,y1,x2,y2,x3,y3...）
        tmp = [] # list，元素为str变量；暂存取样点的坐标集合；
        num_point = 13 # num_point是取样点的数量，点太多labelme加载会很卡
        for index in range(0, len(txt_data), num_point*2): # num_point*2是步长，因为一个点有两个数据（x，y）
            tmp.append(txt_data[index]) #存入x坐标
            tmp.append(txt_data[index + 1]) #存入y坐标
        txt_data = list(map(int, tmp)) # 将tmp的元素从str转成int
        segmentation.append(txt_data)


    # 舍弃轮廓采样点过少的mask
    for item in segmentation:
        if (len(item) > 12): # 大于6个点的mask才被存入

            list1 = [] # 将格式改一下，[x1,y1,x2,y2...]->[[x1,y1], [x2, y2], ...]

            for i in range(0, len(item), 2):
                list1.append([item[i], item[i + 1]])

            seg_info = {'points': list1, "fill_color": 'null', "line_color": 'null', "label": "1",
                        "shape_type": "polygon", "flags": {}}
            coco_output["shapes"].append(seg_info)
    coco_output["imageHeight"] = image.size[1]
    coco_output["imageWidth"] = image.size[0]

    full_path = IMAGE_DIR + name + '.json'
    print(full_path)
    with open(full_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
    #  将原图转换为热力图
    pic_path = IMAGE_DIR + name + '.png'
    picture = cv2.imread(pic_path, cv2.IMREAD_ANYDEPTH)
    heat_img = cv2.applyColorMap(picture, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)  # 将BGR图像转为RGB图像
    cv2.imwrite(IMAGE_DIR + name + '_heat.png', heat_img)
