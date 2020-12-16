import os
import numpy as np
import tensorflow as tf
from PIL import Image
from PIL import ImageFile
from keras.models import model_from_json,load_model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from TOOLS.file_oper import Print

class Classifier:
    def __init__(self,img_width,img_height,model_name):
        self.img_width=img_width
        self.img_height=img_height
        self.model_name=model_name
        self.model=None
    def LoadModel(self):
        if True==os.path.exists(self.model_name):
            self.model=load_model(self.model_name)
    def GetImage(self,paths_arr):
        samples_arr=[]
        effect_paths_arr=[]
        classified_arr = []
        Print('Normal>>准备加载的图像数量：{}!'.format(len(paths_arr)))
        for image_path in paths_arr:
            if True==os.path.exists(image_path):
                mark=-1
                try:
                    im=Image.open(image_path)
                    im_s = np.asarray(im)
                    classified_arr.append(im_s)
                    #classified_arr.append(im)
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
            return effect_paths_arr,Image_arr,classified_arr
        else:
            return None,None
    def PredictImages(self,image_arr):
        preds_result=[]
        preds_confidence=[]
        class_scores=self.model.predict(image_arr,batch_size=image_arr.shape[0])
        for class_score in class_scores:
            pre_class=0
            max_score=0.0
            for n in range(len(class_score)):
                if class_score[n]>max_score:
                    max_score=class_score[n]
                    pre_class=n
            preds_result.append(pre_class)
            preds_confidence.append(max_score)
        return preds_result,preds_confidence
