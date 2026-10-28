import io
import pandas as pd
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

# 定义目标文件夹路径
target_folder = '../data/tiny-imagenet-200/val/'

# 创建一个字典，用来存储每个文件名对应的文件夹类别
val_dict = {}
with open(target_folder + 'val_annotations.txt', 'r') as f:
    # 打开 val_annotations.txt 文件，该文件包含了验证集图片的类别信息
    for line in f.readlines():
        split_line = line.split('\t')  # 每行用 tab 键分割，第一部分是文件名，第二部分是类别
        val_dict[split_line[0]] = split_line[1]  # 将文件名和对应的类别存入字典

# 使用 glob 查找所有验证集图片的路径
paths = glob.glob(target_folder + 'images/*')  # 获取验证集目录下所有图片的路径
paths[0].split('/')[-1]  # 获取第一个文件的文件名，用于检查格式（此处未使用）

# 遍历所有图片路径
for path in paths:
    file = path.split('/')[-1]  # 获取图片文件的文件名
    folder = val_dict[file]  # 根据文件名获取该图片对应的类别
    # 如果目标文件夹不存在，则创建该文件夹
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))  # 创建类别文件夹

# 将所有图片按类别移动到相应的文件夹中
for path in paths:
    file = path.split('/')[-1]  # 获取图片文件的文件名
    folder = val_dict[file]  # 根据文件名获取该图片对应的类别
    dest = target_folder + str(folder) + '/' + str(file)  # 目标路径
    move(path, dest)  # 将图片移动到目标文件夹

# 删除原始的 val_annotations.txt 文件和 images 文件夹
os.remove('../data/tiny-imagenet-200/val/val_annotations.txt')  # 删除类别标注文件
rmdir('../data/tiny-imagenet-200/val/images')  # 删除原始图片文件夹

# 打印完成信息
print('done reformat the validation images')
