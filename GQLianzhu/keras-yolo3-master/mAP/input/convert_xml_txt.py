import xml.etree.ElementTree as ET
import os
from os import getcwd

classes = ['aokeng','qiaopi','zongxiangliewen','hengxiangliewen','bianjiaoliewen',
           'xingxingliewen','zongxianghuashang','hengxianghuashang']  #需要修改为自己的类别
'''
aokeng
qiaopi
zongxiangliewen
hengxiangliewen
bianjiaoliewen
xingxingliewen
zongxianghuashang
hengxianghuashang
'''


def convert_annotation(image_id, list_file):
    xmlpath = 'ground-truth/backup/%s.xml'%(image_id)
    if os.path.isfile(xmlpath):
        in_file = open(xmlpath)
        tree=ET.parse(in_file)
        root = tree.getroot()

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult)==1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            #list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
            list_file.write(str(classes[cls_id]) + " "+ " ".join([str(a) for a in b]) + '\n')
        list_file.close()
wd = getcwd()

dirname="images-optional/" ##该目录为测试照片的存储路径，每次测试照片的数量可以自己设定
path=os.path.join(dirname)
pic_list=os.listdir(path)
#print(len(pic_list))
for filename in pic_list:
    print(filename)
    image_id =filename.split(".")[0]
    if os.path.isfile("ground-truth/"+filename.replace(".bmp",".txt")):
        os.remove("ground-truth/"+filename.replace(".bmp",".txt"))
    list_file = open("ground-truth/"+filename.replace(".bmp",".txt"),'a')
    convert_annotation(image_id,list_file)

