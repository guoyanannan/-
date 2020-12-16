import os
import colorsys
import numpy as np
import cv2

from timeit import default_timer as timer
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from TOOLS.HkjYolo.H_model import yolo_eval, yolo_body, tiny_yolo_body
from TOOLS.HkjYolo.H_utils import letterbox_image

import tensorflow as tf
import os
# 使用第一张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


class YOLO(object):
    _defaults={
            "model_path":'steel_model/steel_model_yolo.h5',
            "anchors_path":'steel_model/steel_anchors.txt',
            "classes_path":'steel_model/steel_classes.txt',
            "score":0.2,
            "iou":0.1,
            "image_size":(416,416),
            "gpu_num":1,
        }
    @classmethod
    def get_defaults(cls,n):
        if n in cls._defaults[n]:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '"+n+"'"
    def __init__(self,**kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)
        self.class_names=self.get_class()
        print(self.class_names)
        self.anchors=self.get_anchors()
        print(self.anchors)
        self.sess=K.get_session()
        self.boxes,self.scores,self.classes=self.generate()
    def get_class(self):
        classes_path=os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names=f.readlines()
        class_names=[c.strip() for c in class_names]
        return class_names
    def get_anchors(self):
        anchors_path=os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors=f.readline()
        anchors=[float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    def generate(self):
        model_path=os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'),"Keras model must be .h5 file."
        num_anchors=len(self.anchors)
        num_classes=len(self.class_names)
        print(model_path)
        is_tiny_version = num_anchors==6
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path)
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == num_anchors/len(self.yolo_model.output) * (num_classes + 5), 'Mismatch between model and given anchor and class sizes'
        print("{} model,anchors,and classes loaded.".format(model_path))
        hsv_tuples=[(x/num_classes,1.,1.) for x in range(num_classes)]
        self.colors=list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        self.colors=list(map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)
        self.input_image_shape=K.placeholder(shape=(2,))
        boxes,scores,classes=yolo_eval(self.yolo_model.output,self.anchors,num_classes,self.input_image_shape,score_threshold=self.score,iou_threshold=self.iou)
        return boxes,scores,classes
    def detect_image(self,image,img_path):
        start=timer()
        if self.image_size != (None,None):
            assert self.image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image=letterbox_image(image,tuple(reversed(self.image_size)))
        else:
            new_image_size=(image.width-(image.width % 32),image.height-(image.height%32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data=np.array(boxed_image,dtype='float32')
        print(image_data.shape)
        image_data/=255.
        image_data=np.expand_dims(image_data,0)
        out_boxes,out_scores,out_classes=self.sess.run([self.boxes,self.scores,self.classes],feed_dict={self.yolo_model.input:image_data,self.input_image_shape:[image.size[1],image.size[0]],K.learning_phase():0})
        print('Found {} boxes for img'.format(len(out_boxes)))
        result=np.asarray(image)
        for i,c in reversed(list(enumerate(out_classes))):
            predicted_class=self.class_names[c]
            box=out_boxes[i]
            score=out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))
            cv2.putText(result,text=label,org=(int((left+right)/2),int((top+bottom)/2)),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(255,0,0),thickness=2)
            cv2.rectangle(result, (left, top), (right, bottom), (0, 255, 0), 2)
            #cv2.namedWindow("result",cv2.WINDOW_NORMAL)
            #cv2.imshow("result",result)
            #cv2.waitKey(0)
            res_path="steel_res_img/"+img_path
            cv2.imwrite(res_path,result)
        end = timer()
        print(end - start)
        return image
    def detect_defect(self,image):
        start=timer()
        if self.image_size != (None,None):
            assert self.image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image=letterbox_image(image,tuple(reversed(self.image_size)))
        else:
            new_image_size=(image.width-(image.width % 32),image.height-(image.height%32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data=np.array(boxed_image,dtype='float32')
        image_data/=255.
        image_data=np.expand_dims(image_data,0)
        out_boxes,out_scores,out_classes=self.sess.run([self.boxes,self.scores,self.classes],feed_dict={self.yolo_model.input:image_data,self.input_image_shape:[image.size[1],image.size[0]],K.learning_phase():0})
        res=[]
        for i,c in reversed(list(enumerate(out_classes))):
            defect=[]
            predicted_class=self.class_names[c]
            box=out_boxes[i]
            score=out_scores[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            defect.append(c)
            defect.append(score)
            defect.append(left)
            defect.append(top)
            defect.append(right)
            defect.append(bottom)
            res.append(defect)
        end = timer()
        print(end - start)
        return res
    def Image_defect_detection(self,image,r=None,c=None):
        start=timer()
        image = Image.fromarray(np.uint8(image))
        if self.image_size != (None,None):
            assert self.image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image=letterbox_image(image,tuple(reversed(self.image_size)))
        else:
            new_image_size=(image.width-(image.width % 32),image.height-(image.height%32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data=np.array(boxed_image,dtype='float32')
        # print('检测数据',image_data)
        # print('检测数据形状',image_data.shape)
        image_data/=255.
        image_data=np.expand_dims(image_data,0)
        out_boxes,out_scores,out_classes=self.sess.run([self.boxes,self.scores,self.classes],feed_dict={self.yolo_model.input:image_data,self.input_image_shape:[image.size[1],image.size[0]],K.learning_phase():0})
        res=[]  #每张图片的每个box的信息
        for i,cls in reversed(list(enumerate(out_classes))):
            defect=[]  #一个box的信息
            predicted_class=self.class_names[cls]
            box=out_boxes[i]
            score=out_scores[i]
            top, left, bottom, right = box
            if r ==None and c == None:
                top = max(0, np.floor(top + 0.5).astype('int32'))    #y1
                left = max(0, np.floor(left + 0.5).astype('int32'))  #x1
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32')) #y2
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))   #x2
                defect.append(cls)
                defect.append(score)
                defect.append(left)
                defect.append(top)
                defect.append(right)
                defect.append(bottom)
                res.append(defect)
            elif r !=None and c !=None:
                top = np.floor(r).astype('int32') + max(0, np.floor(top + 0.5).astype('int32'))  # y1
                left = np.floor(c).astype('int32') + max(0, np.floor(left + 0.5).astype('int32'))  # x1
                bottom = np.floor(r).astype('int32') + min(image.size[1], np.floor(bottom + 0.5).astype('int32'))  # y2
                right = np.floor(c).astype('int32') + min(image.size[0], np.floor(right + 0.5).astype('int32'))  # x2
                defect.append(cls)
                defect.append(score)
                defect.append(left)
                defect.append(top)
                defect.append(right)
                defect.append(bottom)
                res.append(defect)
        end = timer()
        print('time:',end - start)
        print('长度',len(res))
        if len(res) != 0:
            return res  ###返回的是每张图片的所有bbox的信息，二维列表[[],[],[],[]]
            
    def close_session(self):
        self.sess.close()
