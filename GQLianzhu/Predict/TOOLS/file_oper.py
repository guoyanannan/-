import os
import sys
import time
import logging
import datetime
import configparser
from time import strftime,gmtime
from xml.etree import ElementTree as ET


def WriteFile(filename,context):
    fo = open(filename, "w")
    for i in range(len(context)):        
        fo.write(str(context[i]))
        fo.write("\n")
    fo.close()

def DeleteAllFile(path):
    #path=r"E:\ArNTPreClassifiedImage"
    for root,dirs,files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1]=='.bmp':
                sub_file_path = os.path.join(path,file)
                os.remove(sub_file_path)
                Print(sub_file_path)
                
def DeleteFile(FileList):
    for file_path in FileList:
        if True==os.path.exists(file_path):
            os.remove(file_path)

#读类别表
def read_class_table(path):
    if True==os.path.exists(path):
        per=ET.parse(path)
        p=per.findall('./缺陷类别/类别总数')
        classnum=p[0].text
        class_convert_tabel={}
        Print(classnum)
        for i in range(int(classnum)):
            s='./缺陷类别/类别%d' % i
            #print(s)
            p=per.findall(s)
            for oneper in p:
                interal_no=""
                interal_name=""
                exteral_no=""
                for child in oneper.getchildren():
                    #print(child.tag,':',child.text)
                    if child.tag=="内部编号":
                        interal_no=child.text
                    if child.tag=="名称":
                        interal_name=child.text
                    if child.tag=="外部编号":
                        exteral_no=child.text
                class_convert_tabel[interal_no]=exteral_no
        #print(class_convert_tabel)
        return class_convert_tabel
    else:
        input("@@Error：未检测到ArNTClassTable.xml文件，请检查！！！")

def read_class_table_name(path):
    if True==os.path.exists(path):
        per=ET.parse(path)
        p=per.findall('./缺陷类别/类别总数')
        classnum=p[0].text
        class_convert_tabel={}
        class_convert_tabel_1={}
        className_and_exteral = {}
        Print(classnum)
        for i in range(int(classnum)):
            s='./缺陷类别/类别%d' % i
            #print(s)
            p=per.findall(s)
            for oneper in p:
                interal_no=""
                interal_name=""
                exteral_no=""
                for child in oneper.getchildren():
                    #print(child.tag,':',child.text)
                    if child.tag=="内部编号":
                        interal_no=child.text
                    if child.tag=="名称":
                        interal_name=child.text
                    if child.tag=="外部编号":
                        exteral_no=child.text
                class_convert_tabel[interal_name]=interal_no
                class_convert_tabel_1[interal_no]=interal_name
                className_and_exteral[interal_name] = exteral_no
        #print(class_convert_tabel)
        return class_convert_tabel,class_convert_tabel_1,className_and_exteral
    else:
        input("@@Error：未检测到ArNTClassTable.xml文件，请检查！！！")

#读ini配置文件
def read_config_file(path):
    if True==os.path.exists(path):
        conf=configparser.ConfigParser()
        conf.read(path)
        sections = conf.sections()
        class_convert_tabel={}
        classnum=conf.get("Classifier", "ClassNum")
        imgsize=conf.get("Classifier", "ImgSize")
        for i in range(int(classnum)):
            classno="Class%d" % i
            interal_no=conf.get("ClassConversion", classno)
            class_convert_tabel[i]=interal_no
        return class_convert_tabel,imgsize
    else:
        input("Error>>ExternalClassifierInteraction.ini文件不存在，请检查...")
#写ini配置文件     
def write_config_file(classnum,class_interalno,img_size,network):
    conf=configparser.ConfigParser()
    if True==os.path.exists('ExternalClassifierInteraction.ini'):
        os.remove('ExternalClassifierInteraction.ini')
    conf.read("'ExternalClassifierInteraction.ini'")
    conf.add_section("Classifier")
    conf.add_section("ClassConversion")
    conf.set("Classifier", "ClassNum", str(classnum))
    conf.set("Classifier", "ImgSize", str(img_size))
    conf.set("Classifier", "Model", str(network))
    for i in range(classnum):
        classno="Class%d" % i
        conf.set("ClassConversion", classno, str(class_interalno[i]))
    conf.write(open("ExternalClassifierInteraction.ini", "w"))
    
def Print(info):
    curtime=time.strftime("%H:%M:%S")
    loginfo=("%s: %s" % (curtime,info))
    print (loginfo)
    
from keras import backend as K
INTERESTING_CLASS_ID = 1 # Choose the class of interest
def fn(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, INTERESTING_CLASS_ID), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc

def focal_loss(classes_num=None, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)  #全为0的矩阵，shape[batch_size,category_number]
        #shape[batch,category_number],对应one_hot为1的维度值为1-y_hat,其余全部为0
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        #shape[batch,category_number],点乘，对应是某类别的维度值为1-y_hat*log（y_hat），其余全部为0
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))
        balanced_fl = tf.reduce_mean(FT)
        '''
        #2# get balanced weight alpha
        #全为0的矩阵，shape[batch_size,category_number]
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        #shape为[1,classes_num],里面的值为总样本数/每个类别的样本数  dtype：float 32
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #每个类别的的权值
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)  #转换为tensor
        #shape[batch_size,category_number]，每一行的值相同，都是class_w_t2列表里面的值
        classes_weight += classes_w_tensor

        ##shape[batch,category_number],对应one_hot为1的维度值为classes_weight,其余全部为0
        ##此行代码感觉有误，尝试修改
        #alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)
        ##修改-gyn
        alpha = classes_weight

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        #nb_classes = len(classes_num)
        #fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)
        '''
        return balanced_fl#fianal_loss
    return focal_loss_fixed

class Log:
    def __init__(self,log_dir_name):
        self.log_dir_name=log_dir_name
        if False==os.path.exists(self.log_dir_name):
            os.mkdir(self.log_dir_name)
        #self.log_file_name=((self.log_dir_name+"\%s.txt") % time.strftime("%Y%m%d"))
    def AddLog(self,info):
        self.log_file_name=((self.log_dir_name+"\%s.txt") % time.strftime("%Y%m%d"))
        curtime=time.strftime("%H:%M:%S")
        loginfo=("%s: %s" % (curtime,info))
        log=open(self.log_file_name,'a')
        print(loginfo,file=log)
