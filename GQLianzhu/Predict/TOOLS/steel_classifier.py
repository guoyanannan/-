import os
import numpy as np
import cv2

import tensorflow as tf
from PIL import Image
from PIL import ImageFile
from PIL import ImageEnhance
from keras.models import model_from_json,load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from TOOLS.file_oper import Print
from TOOLS.file_oper import fn,focal_loss

class Classifier:
    def __init__(self,img_width,img_height,model_name):
        self.img_width=img_width
        self.img_height=img_height
        self.model_name=model_name
        self.model=None
  
    def LoadModel(self):
        if True==os.path.exists(self.model_name):
            self.model=load_model(self.model_name,custom_objects={'fn': fn,'focal_loss_fixed':focal_loss()})
    def GetMinImage(self,img_list):
        samples_arr=[]
        effect_paths_arr=[]
        Print('Normal>>准备加载的图像数量：{}!'.format(len(img_list)))
        for image in img_list:
            try:
                im=image.convert("RGB")
                im=im.resize((self.img_width,self.img_height))
                im=np.asarray(im,np.float32)
                im=im/255.0
                mark=1
            except:
                Print("Error>>抛出异常!")
                mark=0
                pass
            if mark == 1:
                samples_arr.append(im)
        Print('Normal>>获取到图像数量：{}!'.format(len(samples_arr)))
        if len(samples_arr)>0:
            Image_arr=np.stack(samples_arr,axis=0).astype(np.float32)
            Image_arr=Image_arr.reshape(Image_arr.shape[0],self.img_width,self.img_height,3)
            return effect_paths_arr,Image_arr
        else:
            return None,None
    def GetImage(self,paths_arr):
        samples_arr=[]
        effect_paths_arr=[]
        classified_arr = []   #
        Print('Normal>>准备加载的图像数量：{}!'.format(len(paths_arr)))
        for image_path in paths_arr:
            if True==os.path.exists(image_path):
                mark=-1
                try:
                    im=Image.open(image_path)
                    #im_s = np.asarray(im)
                    #classified_arr.append(im_s)
                    #classified_arr.append(im)
                    #enh_con=ImageEnhance.Contrast(im)
                    #contrast=1.5
                    #im=enh_con.enhance(contrast)
                    im=im.convert("RGB")
                    im=im.resize((self.img_width,self.img_height))
                    im=np.asarray(im,np.float32)
                    im=im/255.0
                    mark=1
                except:
                    Print("Error>>抛出异常!")
                    mark=0
                    pass
                if mark == 1:
                    samples_arr.append(im)
                    effect_paths_arr.append(image_path)
            else:
                Print("Warning>>文件不存在!")
        Print('Normal>>获取到图像数量：{}!'.format(len(samples_arr)))
        if len(samples_arr)>0:
            Image_arr=np.stack(samples_arr,axis=0).astype(np.float32)
            Image_arr=Image_arr.reshape(Image_arr.shape[0],self.img_width,self.img_height,3)
            return effect_paths_arr,Image_arr
        else:
            return None,None
    def GetImagePad(self,paths_arr): # 一个批次或者目录中小于批次数的所有的图片
        samples_arr=[]  #存储每个图片的像素值矩阵ndarray形式
        effect_paths_arr=[]  #存储对应上面每个图片的路径
        Print('Normal>><Padding模式>准备加载的图像数量：{}!'.format(len(paths_arr)))
        for image_path in paths_arr:
            if True==os.path.exists(image_path):
                mark=-1
                try:
                    img = cv2.imread(image_path)
                    if max(img.shape[0]//img.shape[1],img.shape[1]//img.shape[0]) < 7:
                        old_size = img.shape[0:2]
                        target_size = [self.img_height,self.img_width]
                        # ratio = min(float(target_size)/(old_size))
                        ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
                        new_size = tuple([int(i * ratio) for i in old_size])
                        img = cv2.resize(img, (new_size[1], new_size[0]))
                        pad_w = target_size[1] - new_size[1]
                        pad_h = target_size[0] - new_size[0]
                        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
                        left, right = pad_w // 2, pad_w - (pad_w // 2)
                        img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
                    else:
                        img_new = cv2.resize(img,(self.img_height,self.img_width))
                    im = cv2.cvtColor(img_new,cv2.COLOR_BGR2RGB)
                    # print(im.shape)

                    #
                    # im=Image.open(image_path)
                    # im=im.convert("RGB")
                    # im=im.resize((self.img_width,self.img_height))
                    im=np.asarray(im,np.float32)
                    im=im/255.0
                    mark=1
                except:
                    Print("Error>>抛出异常!")
                    mark=0
                if mark == 1:
                    samples_arr.append(im)
                    effect_paths_arr.append(image_path)
            else:
                Print("Warning>>文件不存在!")
        Print('Normal>><Padding模式>获取到图像数量：{}!'.format(len(samples_arr)))
        if len(samples_arr)>0:
            Image_arr=np.stack(samples_arr,axis=0).astype(np.float32)  #shape（B，192，192，3）
            Image_arr=Image_arr.reshape(Image_arr.shape[0],self.img_width,self.img_height,3)
            return effect_paths_arr,Image_arr  #返回的是路径list和shape为（B，h，w，c）的数组
        else:
            return None,None
    def PredictImages(self,image_arr):
        preds_result=[]
        preds_result_top2=[]
        preds_confidence=[]
        preds_confidence_top2=[]
        class_scores=self.model.predict(image_arr,batch_size=image_arr.shape[0])
        for class_score in class_scores:
            # pre_class=0
            # pre_class_top2=0
            # max_score=0.0
            # top2_score = 0.0
            class_score_copy = list(class_score)
            max_numbers = []
            max_indexs = []
            for _ in range(2):
                max_score = max(class_score_copy)
                max_index = class_score_copy.index(max_score)
                class_score_copy[max_index] = 0
                max_numbers.append(max_score)
                max_indexs.append(max_index)
            # for n in range(len(class_score)):
            #     if class_score[n]>max_score:
            #         max_score=class_score[n]
            #         pre_class=n
            preds_result.append(max_indexs[0])
            # preds_result_top2.append(pre_class_top2)
            preds_confidence.append(max_numbers[0])
            preds_result_top2.append(max_indexs[1])
            preds_confidence_top2.append(max_numbers[1])
            Print('Normal>>一个批次的预测结果解析完成')
        return preds_result,preds_result_top2,preds_confidence
