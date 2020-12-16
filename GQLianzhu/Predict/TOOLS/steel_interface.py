import sys
from TOOLS.steel_yolo import YOLO
from PIL import Image
import os
import cv2
import numpy as np

def DetectDirAllImg(yolo,path):
    for _,_,filenames in os.walk(path):
        for filename in filenames:
            img_path=os.path.join(path,filename)
            image = Image.open(img_path)
            image_s=image.split()
            print(len(image_s))
            if len(image_s) != 1:
                image = image.convert('RGB')
            r_image = yolo.detect_image(image,filename)

def DefectDetectInterface(yolo,img_path_list,path=None):  #实例化后的yolo对象
    res_defect=[]
    for img_path in img_path_list:
        image = Image.open(img_path)
        image_s=image.split()
        if len(image_s) != 1:
            image = image.convert('RGB')
        res = yolo.detect_defect(image)
        res_defect.append(res)
        ##保存图片
        if None != path:
            result=np.asarray(image)
            for i in range(len(res)):
                cls,score, left, top, right, bottom = res[i]
                cv2.putText(result,text="ss",org=(int((int(left)+int(right))/2),int((int(top)+int(bottom))/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
                cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)    
                if False==os.path.exists(path):
                    os.mkdir(path)
                name=img_path.split('\\')
                res_path=path+name[len(name)-1]
                cv2.imwrite(res_path,result)
        ##
    return res_defect
def yolo_Interface(yolo,img_path_list,win_size=1024,stride=768,path=None):  #实例化后的yolo对象
    import cv2

    cls_name = yolo.get_class()
    every_img_result = [] #每张图片的nms后的框的信息
    for img_path in img_path_list:
        res_defect = []
        img= cv2.imread(img_path)
        #ImageList.append(img)  #原图
        res = yolo.Image_defect_detection(img) #每张图片所有框的list=[[],[],[]]
        #print(res)
        res_defect.append(res)
        h, w, _ = img.shape
        # win_size = 1024 ,stride =768
        for r in range(0, (h - win_size) + 1, stride):  # H方向进行切分
            for c in range(0, (w - win_size) + 1, stride):  # W方向进行切分
                tmp = img[r: r + win_size, c: c + win_size]  #切图
                #print(tmp.shape)
                #ImageList.append(tmp)
                res = yolo.Image_defect_detection(tmp,r,c)    #每张图的左边转换后的每个bbox信息[[],[],[]....]
                #print(res)
                res_defect.append(res)
        # print('删除前',res_defect)
        res_defect = [x for x in res_defect if x != None]
        #print('删除后',res_defect)
        res_defect_list = []
        for ress in res_defect:
            for res in ress:
                res_defect_list.append(res)
        #res_defect_arr = np.asarray(res_defect)
        #print(res_defect_arr.shape)
        #res_defect_list = res_defect_arr.reshape(-1,res_defect_arr.shape[2])
        print('NMS前',res_defect_list)
        # print(res_defect_list.tolist())

        ####----将整合的框进行NMS----####
        if len(res_defect_list) != 0:
            from TOOLS.Steel_NMS import SteelNMS
            BBox_after_NMS = SteelNMS(res_defect_list,cls_name,0.4)
            every_img_result.append(BBox_after_NMS)
            ##保存图片
            if None != path:
            	if False == os.path.exists(path):
                    os.mkdir(path)
                if len(os.listdir(path)) <= 1000:
                    result = np.asarray(img)
                    for i in range(len(BBox_after_NMS)):
                        cls, score, left, top, right, bottom = BBox_after_NMS[i]
                        print('NMS后',cls, score, left, top, right, bottom)
                        ss = cls_name[int(cls)]
                        cv2.putText(result,text=ss,org=(int((int(left)+int(right))/2),int((int(top)+int(bottom))/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
                        cv2.rectangle(result, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                        name = img_path.split('\\')
                        res_path = path + name[len(name) - 1]
                        cv2.imwrite(res_path, result)
        else:
            every_img_result = res_defect_list
            if None != path:
                if False == os.path.exists(path):
                    os.mkdir(path)
                if len(os.listdir(path)) <= 1000:
                    result = np.asarray(img)
                    name = img_path.split('\\')
                    res_path = path + name[len(name) - 1]
                    cv2.imwrite(res_path, result)
    return every_img_result,cls_name

def GetImgPathListFromDir(path):
    img_path_list=[]
    for _,_,filenames in os.walk(path):
        for filename in filenames:
            img_path=os.path.join(path,filename)
            img_path_list.append(img_path)
    return img_path_list




if __name__ == '__main__':
    yolo=YOLO()
    path='steel_test_img'
    DetectDirAllImg(yolo,path)
    
##    img_path_list=GetImgPathListFromDir(path)
##    res_defect=DefectDetectInterface(yolo,img_path_list)
##    print(res_defect)
    
    while True:
        img = input('Input image filename:')
        try:
            img_path=os.getcwd()+'\\'+img
            print(img_path)
            image = Image.open(img_path)
            r_image = yolo.detect_image(image,img)
        except:
            print('Open Error! Try again!')
            continue
            
    YOLO.close_session()


