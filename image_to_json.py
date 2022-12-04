import os
import cv2
import json
import numpy as np
from tqdm import tqdm

# KolektorSDD2专用，直接删除，注意拷贝
def delete_not_label_images(imgs_path):
    for label_name in tqdm(os.listdir(imgs_path)):
        if 'GT' not in label_name:
            label_path = imgs_path + label_name
            os.remove(label_path)

# KolektorSDD2专用，直接删除，注意拷贝
def delete_label_images(imgs_path):
    for label_name in tqdm(os.listdir(imgs_path)):
        if 'GT' in label_name:
            label_path = imgs_path + label_name
            os.remove(label_path)

# 要确保标签图片是二值化图片，即值只有0和255
def show_images_unique_values(imgs_path):
    label_list = []
    for label_name in tqdm(os.listdir(imgs_path)):
        label_path = imgs_path + label_name
        img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        key = np.unique(img)
        list = key.tolist()
        for label in list:
            if label not in label_list:
                label_list.append(label)
    print(label_list)

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

def RSDDs():
    imgs_path1 = './Type-I RSDDs dataset/GroundTruth/'
    jsons_path1 = './Type-I RSDDs dataset/jsons/'
    imgs_path2 = './Type-II RSDDs dataset/GroundTruth/'
    jsons_path2 = './Type-II RSDDs dataset/jsons/'
    label_name = 'defect'
    images_to_jsons(imgs_path1, jsons_path1, label_name)
    images_to_jsons(imgs_path2, jsons_path2, label_name)

def KolektorSDD1():
    imgs_path = './image_labels/'
    jsons_path = './jsons/'
    label_name = 'defect'
    images_to_jsons(imgs_path, jsons_path, label_name)

def KolektorSDD2():
    imgs_path1 = './train/'
    imgs_path2 = './test/'
    jsons_path = './jsons/'
    label_name = 'defect'
    delete_not_label_images(imgs_path1)
    delete_not_label_images(imgs_path2)
    images_to_jsons(imgs_path1, jsons_path, label_name)
    images_to_jsons(imgs_path2, jsons_path, label_name)

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
    pass
