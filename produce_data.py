import cv2
import numpy as np
import os
#切割图片，图片文件夹中不能含有其他文件（如json、txt）
file_path = "./img"
height = 224
width = 224
# 遍历目标图像路径
# global root, dirs, files, ssdna_names
directory = list(os.walk(file_path))
root = directory[0][0]
files = directory[0][2]
ssdna_names = []
for name in files:
    if name.split('.')[0][-5:] != 'masks':
        name = name.split('.')[0]
        ssdna_names.append(name)

# 创建不同的id文件夹以便后续分割后的图像存放
for id in ssdna_names:
    folder = os.path.exists(root+'//'+id)
    if not folder:
        os.mkdir(root+'//'+id)

dirs = directory[0][1]

# 对file_path下的所有图像进行分割
for pic_name in files:
    # 分割的图片的位置,图片保存位置
    pic_path = root + '//' + pic_name
    pic_target = root + '//' + pic_name[:9]# 存放分割数据的文件夹
    # 要分割的子图尺寸
    cut_height = height
    cut_width = width

    # 读取要分割的图片，以及其尺寸等数据,cv2.IMREAD_ANYDEPTH可以读进灰度图
    picture = cv2.imread(pic_path, cv2.IMREAD_ANYDEPTH)
    (pic_height, pic_width) = picture.shape

    # 计算能够被cut_height, cut_width整除的填充pixel数量
    difference_h = cut_height - (pic_height % cut_height)
    difference_w = cut_width - (pic_width % cut_width)

    # 对原图进行填充（仅对difference_h，difference_w都为偶数的情况有效）
    after_pad = np.pad(picture,
                       ((int(difference_h), 0),
                        (int(difference_w), 0)),
                       'constant', constant_values=-0)
    # cv2.imwrite('111.png', after_pad) #在该脚本所在路径保存填充后的图像
    # 计算可以划分的横纵的个数
    (after_pad_height, after_pad_width) = after_pad.shape
    num_height = int(after_pad_height / cut_height) * 2 - 1
    num_width = int(after_pad_width / cut_width) * 2 - 1
    step_length = int(cut_width / 2)# 以步长为剪裁框大小的一半进行滑动剪裁
    # for循环迭代生成
    for h in range(0, num_height):
        for w in range(0, num_width):
            h_start_pix = h * step_length
            w_start_pix = w * step_length
            pic = after_pad[h_start_pix: h_start_pix + cut_height,
                  w_start_pix: w_start_pix + cut_width
                  ]
            if 'masks' in pic_name:
                result_path = pic_target + '//' + '{}_{}_{}_{}.png'.format(pic_name.split('.')[0][:9],
                                                                           h + 1, w + 1, 'masks')
            else:
                result_path = pic_target + '//' + '{}_{}_{}.png'.format(pic_name.split('.')[0], h + 1, w + 1)

            cv2.imwrite(result_path, pic)
    print("{} is done".format(pic_name))

# 开始删除label中没有mask的切割图像
directory = list(os.walk(root))
dir_list = directory[0][1]
for dir in dir_list:
    dir_path = root + '//' + dir
    directory_seg = list(os.walk(dir_path))
    root_seg = directory_seg[0][0]
    files_seg = directory_seg[0][2]
    delete_id = []
    for id in files_seg:
        path_seg = root_seg + '//' + id
        if 'masks' in id:
            picture_seg = cv2.imread(path_seg, cv2.IMREAD_ANYDEPTH)
            if picture_seg.max() == 0:
                name_list = id.split('.')[0].split('_')
                name_list.remove('masks')
                name = '_'.join(name_list)
                delete_id.append(name)
    for name in delete_id:
        mask_path = root_seg + '//' + name + "_masks.png"
        ssdna_path = root_seg + '//' + name + ".png"
        os.remove(mask_path)
        os.remove(ssdna_path)