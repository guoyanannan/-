#把图片切块并且得到xml文件


import sys
import os
import cv2
import numpy as np
import os.path
import shutil
from xml.dom.minidom import Document
from tqdm import tqdm
if sys.version_info[0] ==2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def check_file(Name):
    if os.path.exists(Name):
        shutil.rmtree(Name)
        os.makedirs(Name)
    else:
        os.makedirs(Name)
def clip_img(No, oriname):  #(index,file_name)
    from_name = os.path.join(origin_dir, oriname+'.bmp')
    img = cv2.imread(from_name,cv2.IMREAD_UNCHANGED)
    print(img.shape)
    h_ori,w_ori =img.shape#保存原图的大小

    #img = cv2.resize(img, (2048, 2048))#可以resize也可以不resize，看情况而定
    h, w= img.shape
    xml_name = os.path.join(annota_dir, oriname+'.xml')#读取每个原图像的xml文件
    print(xml_name)
    if os.path.exists(xml_name):
        xml_ori = ET.parse(xml_name).getroot()
        res = np.empty((0,5))#存放坐标的四个值和类别
        for obj in xml_ori.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')  #旗下有四个子节点pt
            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []  #当前xml坐标框的信息
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = int(cur_pt*h/h_ori) if i%2==1 else int(cur_pt * w / w_ori)  #取余
                bndbox.append(cur_pt)
            #label_idx = self.class_to_ind[name]
            bndbox.append(name)
            res = np.vstack((res, bndbox))  #纵向堆叠每个目标框信息即一行行罗列
        i = 0
        win_size = 1024#分块的大小

        stride = 1024-256#重叠的大小（768），设置这个可以使分块有重叠  stride =win_size 说明设置的分块没有重叠
        for r in range(0, (h - win_size)+1, stride): # H方向进行切分
            for c in range(0, (w - win_size)+1, stride): # W方向进行切分
                flag = np.zeros([1, len(res)])  ## 修改flag = np.zeros([1, len(res)])
                youwu = False    #是否有物体
                xiefou = True    #是否记录
                tmp = img[r: r+win_size, c: c+win_size]
                i = i + 1
                print('r:',r,'c',c,tmp.shape)
                for re in range(res.shape[0]):
                    xmin,ymin,xmax,ymax,label = res[re]
                    ###---如果框完全在切分得小图中，进行左边转换，否则只有一小部分的直接过滤---###
                    ###---如果有其它情况，比如框的坐标不准确，直接跳出整个循环---###
                    if int(xmin)>=c and int(xmax) <=c+win_size and int(ymin)>=r and int(ymax)<=r+win_size:
                        flag[0][re] = 1  #用于判断是第几个bbox坐标信息在该小图中
                        youwu = True
                    elif int(xmin)<c or int(xmax) >c+win_size or int(ymin) < r or int(ymax) > r+win_size:
                        pass
                    else:
                        xiefou = False
                        break;
                ###---只有框完全在某个小图中时才进行坐标转换---###
                ###---如果有非常大的缺陷横跨两张图那么就是属于大目标，放进去原图即可---###
                if xiefou:#如果物体被分割了，则忽略不写入
                    if youwu:#有物体则写入xml文件
                        doc = Document()
                        annotation = doc.createElement('annotation')
                        doc.appendChild(annotation)
                        for re in range(res.shape[0]):
                            xmin,ymin,xmax,ymax,label = res[re]
                            xmin=int(xmin)
                            ymin=int(ymin)
                            xmax=int(xmax)
                            ymax=int(ymax)
                            if flag[0][re] == 1:
                                xmin=str(xmin-c)
                                ymin=str(ymin-r)
                                xmax=str(xmax-c)
                                ymax=str(ymax-r)
                                #创建annotation下的子节点 object
                                object_charu = doc.createElement('object')
                                annotation.appendChild(object_charu)
                                #创建annotation下的子节点name  并写入文本内容
                                name_charu = doc.createElement('name')
                                name_charu_text = doc.createTextNode(label)
                                name_charu.appendChild(name_charu_text)
                                object_charu.appendChild(name_charu)
                                #创建annotation下的子节点difficult  并写入文本内容
                                dif = doc.createElement('difficult')
                                dif_text = doc.createTextNode('0')
                                dif.appendChild(dif_text)
                                object_charu.appendChild(dif)
                                # 创建annotation下的子节点bndbox
                                bndbox = doc.createElement('bndbox')
                                object_charu.appendChild(bndbox)
                                #创建bndbox下的子节点xmin，创建xmin真实值的节点，将节点添加到父节点
                                xmin1 = doc.createElement('xmin')
                                xmin_text = doc.createTextNode(xmin)
                                xmin1.appendChild(xmin_text)
                                bndbox.appendChild(xmin1)
                                #ymin节点
                                ymin1 = doc.createElement('ymin')
                                ymin_text = doc.createTextNode(ymin)
                                ymin1.appendChild(ymin_text)
                                bndbox.appendChild(ymin1)
                                #xmax节点
                                xmax1 = doc.createElement('xmax')
                                xmax_text = doc.createTextNode(xmax)
                                xmax1.appendChild(xmax_text)
                                bndbox.appendChild(xmax1)
                                #ymax节点
                                ymax1 = doc.createElement('ymax')
                                ymax_text = doc.createTextNode(ymax)
                                ymax1.appendChild(ymax_text)
                                bndbox.appendChild(ymax1)
                            else:
                                continue
                        print('bbox坐标转换完毕')
                        xml_name = oriname+'_%05d.xml' % (i)
                        to_xml_name = os.path.join(target_dir2, xml_name)

                        with open(to_xml_name, 'wb+') as f:
                            f.write(doc.toprettyxml(indent="\t", encoding='utf-8'))
                        #name = '%02d_%02d_%02d_.bmp' % (No, int(r/win_size), int(c/win_size))
                        img_name = oriname+'_%05d.bmp' %(i)
                        to_name = os.path.join(target_dir1, img_name)
                        # i = i+1
                        cv2.imwrite(to_name, tmp)


#######--------------------------------------------------------------#######################

if __name__ == '__main__':
    add = 0
    bdd = 0

    target_dir1 = '../SaveDir_Total/JPGImages_img/'
    target_dir2 = '../SaveDir_Total/Annotations_xml/'
    path = '../DataOrigin'
    check_file(target_dir2)
    check_file(target_dir1)

    for dirname in os.listdir(path):
        origin_dir = os.path.join(path,dirname)
        annota_dir = origin_dir
        print(origin_dir)
        try:
            for No, name in tqdm(enumerate(os.listdir(origin_dir))):
                if name.split('.')[-1] == 'bmp':
                    print(No, name,name.rstrip('.bmp'))
                    add +=1
                    img_path = os.path.join(origin_dir,name)
                    img_mv_path = os.path.join(target_dir1,name)
                    shutil.copy(img_path,img_mv_path)
                    clip_img(No, name.rstrip('.bmp'))
                if name.split('.')[-1] == 'xml':
                    bdd +=1
                    xml_path = os.path.join(origin_dir,name)
                    xml_mv_path = os.path.join(target_dir2,name)
                    shutil.copy(xml_path, xml_mv_path)
        except:
            print(f'遍历文件夹{dirname}出错')
            continue