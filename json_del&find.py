# -*- coding: utf-8 -*-
# @Time : 2022/5/5 10:30
# @Author : Chenglong She
# @File : json_del&find.py
# @Software: PyCharm
# 专门针对mask2json.py生成的异常json文件，cv.findContours()函数很有可能返回轮廓点小于三个的json文件，若直接用来生成coco数据集会造成个别
# 标签图片没有mask或者标签图片中个别mask的轮廓点数量为1或者2(小于3个点无法围成多边形，无法生成mask)，这将导致网络出现无法训练的bug并且该bug
# 比较难发现，一定要好好检查数据集是否存在问题再训练。 该脚本会删除json文件中轮廓点为一个或两个的mask。 会print不存在mask的json文件名，需要
# 手动删除print出来的json文件。

import json
import os

def fause_json(jsonfilelist,json_path,fause_json_list,errorjson):
    '''
    筛选因为形状问题无法转化的 json 文件
    '''
    for i in jsonfilelist:
        f=open(json_path+'/'+i, "r")
        jsonfile=json.loads(f.read())
        if jsonfile['shapes']==[]:
            errorjson.append(i)
        for index, j in enumerate(jsonfile['shapes']):
            point_list = j['points']
            if len(point_list)<3:
                fause_json_list.append(i)
    return fause_json_list, errorjson


dict = {}


# 用来存储数据

def get_json_data(json_path):
    # 获取json里面数据

    with open(json_path, 'rb') as f:
        # 定义为只读模型，并定义名称为f

        params = json.load(f)
        # 加载json文件中的内容给params

        for index, j in enumerate(params['shapes']):
            params['shapes'][index]['label'] = str(index+1)
            point_list = j['points']
            if len(point_list) < 3:
                # fause_json_list.append(i)
                params['shapes'].remove(j)
        # 修改内容

        # print("params", params)
        # # 打印

        dict = params
        # 将修改后的内容保存在dict中

    f.close()
    # 关闭json读模式

    return dict
    # 返回dict字典内容


def write_json_data(dict, json_path):
    # 写入json文件

    with open(json_path, 'w') as r:
        # 定义为写模式，名称定义为r

        json.dump(dict, r)
        # 将dict写入名称为r的文件中

    r.close()
    # 关闭json写模式

IMAGE_DIR = '/jdfssz1/ST_SUPERCELLS/P21Z10200N0171/USER/shechenglong/mask_rcnn/tool/data'
directory = list(os.walk(IMAGE_DIR))
root = directory[0][0]
files = directory[0][2]
jsonfilelist = []
errorjson = []
fause_json_list = []
jsonpath = IMAGE_DIR
for i in files:
    if 'json' in i:
        jsonfilelist.append(i)
fause_json_list = fause_json(jsonfilelist, jsonpath, fause_json_list, errorjson)
print(errorjson)
for i in jsonfilelist:
        json_path = IMAGE_DIR + '/'+i
        the_revised_dict = get_json_data(json_path)
        write_json_data(the_revised_dict, json_path)

