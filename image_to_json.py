# 用于将语义分割的灰度图像标签转换成实例分割的labelme格式的json标签
# 一张标签图片上的目标类别只能有一类，多类区分不出来，且目标类别像素值应为255，或大于二值化阈值的像素值
# 需要特别注意：
# data['imagePath'] = img_path.split('/')[-1][:-3] + 'jpg'
# json_path = jsons_path + image_name[:-3] + 'json'
# 以上这两行可能需要小修改，均为名字格式相关的修改

import os
import cv2
import json
import numpy as np
from tqdm import tqdm

def one_image_to_json(img_path, json_path, label_name):
    null = None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    # 若不是二值化图像，在此二值化
    retval, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 若目标太小，就膨胀一下
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    # img = cv2.dilate(img, kernel)  # 膨胀操作

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return

    # cv2.imshow("image", img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # 按顺序构建json字典
    data = {}
    data['version'] = '5.0.1'
    data['flags'] = {}
    # 处理shapes信息
    data['shapes'] = []
    for contour in contours:
        shape = {}
        shape['label'] = label_name
        contour = contour.reshape(contour.shape[0], -1)
        contour = contour.astype(float)
        contour = contour.tolist()
        if len(contour) < 3:
            continue
        shape['points'] = contour
        shape['group_id'] = null
        shape['shape_type'] = 'polygon'
        shape['flags'] = {}
        data['shapes'].append(shape)
    if len(data['shapes']) == 0:
        return
    # 处理剩余信息
    data['imagePath'] = img_path.split('/')[-1][:-3] + 'jpg' #########这里需要特别注意，可能需要微调
    data['imageData'] = null
    data['imageHeight'] = height
    data['imageWidth'] = width
    # 写入json文件
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)

def images_to_jsons(imgs_path, jsons_path, label_name):
    for image_name in tqdm(os.listdir(imgs_path)):
        img_path = imgs_path + image_name
        json_path = jsons_path + image_name[:-3] + 'json'  #############这里需要特别注意，可能需要微调
        one_image_to_json(img_path, json_path, label_name)

# 将RSDDs数据集图片标签转换成labelme的json标签
def RSDDs():
    imgs_path1 = './Type-I RSDDs dataset/GroundTruth/'
    jsons_path1 = './Type-I RSDDs dataset/jsons/'
    imgs_path2 = './Type-II RSDDs dataset/GroundTruth/'
    jsons_path2 = './Type-II RSDDs dataset/jsons/'
    label_name = 'defect'
    images_to_jsons(imgs_path1, jsons_path1, label_name)
    images_to_jsons(imgs_path2, jsons_path2, label_name)

# 将KolektorSDD数据集图片标签转换成labelme的json标签
# https://www.vicos.si/resources/kolektorsdd/
def KolektorSDD1():
    imgs_path = './image_labels/'
    jsons_path = './jsons/'
    label_name = 'defect'
    images_to_jsons(imgs_path, jsons_path, label_name)

# 将KolektorSDD2数据集图片标签转换成labelme的json标签
# https://www.vicos.si/resources/kolektorsdd2/
def KolektorSDD2():
    imgs_path1 = './train/'
    imgs_path2 = './test/'
    jsons_path = './jsons/'
    label_name = 'defect'
    delete_not_label_images(imgs_path1)
    delete_not_label_images(imgs_path2)
    images_to_jsons(imgs_path1, jsons_path, label_name)
    images_to_jsons(imgs_path2, jsons_path, label_name)

# 将Magnetic Tile数据集图片标签转换成labelme的json标签
# https://github.com/Charmve/Surface-Defect-Detection/tree/master/Magnetic-Tile-Defect
def Magnetic_Tile():
    MT_Blowhole_path = './MT_Blowhole/Imgs/'
    MT_Break_path = './MT_Break/Imgs/'
    MT_Crack_path = './MT_Crack/Imgs/'
    MT_Fray_path = './MT_Fray/Imgs/'
    MT_Uneven_path = './MT_Uneven/Imgs/'
    jsons_path = './jsons/'
    images_to_jsons(MT_Blowhole_path, jsons_path, 'Blowhole')
    images_to_jsons(MT_Break_path, jsons_path, 'Break')
    images_to_jsons(MT_Crack_path, jsons_path, 'Crack')
    images_to_jsons(MT_Fray_path, jsons_path, 'Fray')
    images_to_jsons(MT_Uneven_path, jsons_path, 'Uneven')

if __name__ == '__main__':
    RSDDs()
