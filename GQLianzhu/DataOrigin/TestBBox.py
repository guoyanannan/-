# -*- coding: utf-8 -*-
#批量处理img和xml文件，根据xml文件中的坐标把img中的目标标记出来，并保存到指定文件夹，方便自己查看目标标记的是否准确。
import xml.etree.ElementTree as ET
import os, cv2
from SplitImg_and_RewriteXML import check_file
from tqdm import tqdm

annota_dir = '../SaveDir_Total/Annotations_xml'
origin_dir = '../SaveDir_Total/JpGImages_img'
target_dir1='../SaveDir_Total/Check_the_box'

def divide_img(oriname):
    img_file = os.path.join(origin_dir, oriname + '.bmp')
    im = cv2.imread(img_file,cv2.IMREAD_UNCHANGED)
    xml_file = os.path.join(annota_dir, oriname + '.xml')  # 读取每个原图像的xml文件
    tree = ET.parse(xml_file)
    root = tree.getroot()
    #im = cv2.imread(imgfile)
    for object in root.findall('object'):
        object_name = object.find('name').text
        Xmin = int(object.find('bndbox').find('xmin').text)
        Ymin = int(object.find('bndbox').find('ymin').text)
        Xmax = int(object.find('bndbox').find('xmax').text)
        Ymax = int(object.find('bndbox').find('ymax').text)
        # color = (4, 250, 7)
        color = (255, 0, 0)
        cv2.rectangle(im, (Xmin, Ymin), (Xmax, Ymax), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, object_name, (Xmin, Ymin - 7), font, 0.5, (6, 230, 230), 2)
        cv2.imshow('01', im)
        cv2.waitKey(1)

    img_name = oriname + '.bmp'
    to_name = os.path.join(target_dir1, img_name)
    cv2.imwrite(to_name, im)


if __name__ == '__main__':
    check_file(target_dir1)

    img_list = os.listdir(origin_dir)
    for name in img_list:
        divide_img(name.rstrip('.bmp'))