################################################################################
##########################         BKVision        #############################
##########################       wu kun peng       #############################
##########################       version 3.0       #############################
################################################################################

#加载keras模型.h5模型
import os
import sys
import math
import logging
import time
import numpy as np
from PIL import Image
import tensorflow as tf
import datetime
import threading
from multiprocessing import Process
from multiprocessing import Pool
from xml.etree import ElementTree as ET
import configparser
from configparser import ConfigParser
from TOOLS.steel_yolo import YOLO
from TOOLS.steel_interface import DefectDetectInterface,yolo_Interface
from TOOLS.db_oper import ConnectDB,CreateAllCamProcedure,WriteDatabase
from TOOLS.file_oper import DeleteAllFile,DeleteFile,read_class_table,read_config_file,Print,Log,read_class_table_name
from TOOLS.steel_classifier import Classifier
import shutil
#ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
# 使用第一张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras import backend as K
config = tf.ConfigProto(allow_soft_placement = True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.7)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

###**************************现场需要检测的缺陷类别****************************###
#g_EnableClassDetectNo=(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34)
g_DefectClassConfig={"0":{"enable_detect":0,"class_conf":1},
                     "1":{"enable_detect":1,"class_conf":0.6},
                     "2":{"enable_detect":1,"class_conf":0.6},
                     "3":{"enable_detect":1,"class_conf":0.6},
                     "4":{"enable_detect":1,"class_conf":0.6},
                     "5":{"enable_detect":1,"class_conf":0.6},
                     "6":{"enable_detect":1,"class_conf":0.6},
                     "7":{"enable_detect":1,"class_conf":0.6},
                     "8":{"enable_detect":1,"class_conf":0.6},
                     "9":{"enable_detect":1,"class_conf":0.6},
                     "10":{"enable_detect":1,"class_conf":0.6},
                     "11":{"enable_detect":1,"class_conf":0.6},
                     "12":{"enable_detect":1,"class_conf":0.6},
                     "13":{"enable_detect":1,"class_conf":0.6},
                     "14":{"enable_detect":1,"class_conf":0.6},
                     "15":{"enable_detect":1,"class_conf":0.6},
                     "16":{"enable_detect":1,"class_conf":0.6},
                     "17":{"enable_detect":1,"class_conf":0.6},
                     "18":{"enable_detect":1,"class_conf":0.6},
                     "19":{"enable_detect":1,"class_conf":0.6},
                     "20":{"enable_detect":1,"class_conf":0.6},
                     "21":{"enable_detect":1,"class_conf":0.6},
                     "22":{"enable_detect":1,"class_conf":0.6},
                     "23":{"enable_detect":1,"class_conf":0.6},
                     "24":{"enable_detect":1,"class_conf":0.6},
                     "25":{"enable_detect":1,"class_conf":0.6},
                     "26":{"enable_detect":1,"class_conf":0.6},
                     "27":{"enable_detect":1,"class_conf":0.6},
                     "28":{"enable_detect":1,"class_conf":0.6},
                     "29":{"enable_detect":1,"class_conf":0.6},
                     "30":{"enable_detect":1,"class_conf":0.6},
                     "31":{"enable_detect":1,"class_conf":0.6},
                     "32":{"enable_detect":1,"class_conf":0.6}}
#input(g_DefectClassConfig['1']["class_conf"])
###**************************现场需要精确定位的类别****************************###
g_EnableClassLocation=()
###**************************现场数据库IP地址****************************###
g_DefectDBIP="192.168.0.100"
###**************************现场相机数量****************************###
g_CameraNum=4
###**************************现场待处理缺陷图片存储路径****************************###
# g_DefectImgPath=r"D:\ArNTPreClassifiedImage"
g_DefectImgPath=r"D:\ArNTPreClassifiedAllImage"
#是否使用分类算法
IsUseClsArith = False
#是否保存样本
IsSave = False
#是否屏蔽头部、边部
IsShieldTop = False
ShieldTop = 30000
IsShieldEdge = False
ShieldEdge = 150
#判断窄板2号or3号++相机成像是否满足屏蔽边部的阈值（阈值：窄板背景区域X轴长度）
EdgeNoDetectPixel = 70

def SuperVision():
    log_s=Log("./Log_SDC")
    while(1):
        global g_error_s
        index_begin=g_index_begin
        time.sleep(10)#5
        index_end=g_index_end
        if index_begin>index_end:
            g_error_s+=1
            Print('Wraning>>当前存在大于5s检测缓慢有{}次!'.format(g_error_s))
            log_s.AddLog('Wraning>>当前存在大于5s检测缓慢有{}次!'.format(g_error_s))
        if g_error_s>20:#10
            Print("Error>>分类器程序可能停止工作，程序即将关闭！")
            log_s.AddLog("Error>>分类器程序可能停止工作，程序即将关闭！")
            os.system('taskkill /IM ArNTSteelDefectsClassifier.exe /F')




def Main():
    log=Log("./Log_SDC")
    log.AddLog("Normal>>开始运行程序！")
    ##----使用分类模型需要的变量----###
    tabel1=read_class_table('steel_model/ArNTClassTable_ASCII_cls.xml')  #inter:exteral
    tabel2,img_size=read_config_file("steel_model/ExternalClassifierInteraction_cls.ini")  #clsindex:inter

    #name:exteral
    _,_,classname_exteral = read_class_table_name('steel_model/ArNTClassTable_ASCII_detect.xml')
    # Print(tabel1)
    # Print(tabel2)
    Print(classname_exteral)
    Print(g_DefectClassConfig)
    #创建存储过程

    CreateAllCamProcedure(g_CameraNum,g_DefectDBIP)
    Print("Normal>>成功创建存储过程！")
    log.AddLog("Normal>>成功创建存储过程！")

    #设置每批次的数量
    CameraNum=g_CameraNum
    # batch_size=64
    batch_size=1  #-->需要切分，实际6
    path=g_DefectImgPath   #"D:\ArNTPreClassifiedImage"

    #清除缓存图像
    DeleteAllFile(path)



    #初始化模型
    classifier=Classifier(int(img_size),int(img_size),"steel_model/steel_model_classifier.h5")
    classifier.LoadModel()
    Print("Normal>>成功载入模型!")
    log.AddLog("Normal>>成功载入模型!")
    #该变量是累加批次数量
    g_index=0
    #初始化yolo模型 arvg方便传参
    arvg={ 'model_path': 'steel_model/steel_model_yolo.h5', 'anchors_path': 'steel_model/steel_anchors.txt', 'classes_path': 'steel_model/steel_classes.txt',"image_size":(416,416)}
    yolo=YOLO(**arvg)
    #监督进程
    global g_index_begin
    global g_index_end
    global g_error_s
    g_index_begin=0
    g_index_end=0
    g_error_s=0

    mt = threading.Thread(target=SuperVision,args=())
    mt.start()


    '''
    ###----后续使用----###
    #判断哪个相机是边部
    top_L_ini = 1 #初始
    top_L = 1 #变量
    top_R_ini = CameraNum//2
    top_R = CameraNum//2
    bot_L_ini = CameraNum//2 + 1
    bot_L = CameraNum//2 + 1
    bot_R_ini = CameraNum
    bot_R = CameraNum
    '''
    global DefectList
    while(1):
        #遍历文件夹，获取到一个批次要处理的图像文件
        DefectAllPathNameSet=[] 
        for root,dirs,files in os.walk(path):
            if len(files)<=0:
                # Print("Warning>>没有需要分类的数据啦！")
                # log.AddLog("Warning>>没有需要分类的数据啦！")
                Print("Warning>>没有需要检测的数据啦！")
                log.AddLog("Warning>>没有需要检测的数据啦！")
                time.sleep(1)
            for file in files:
                if os.path.splitext(file)[1]=='.bmp':
                    sub_file_path = os.path.join(path,file)
                    DefectAllPathNameSet.append(sub_file_path)
                    if len(DefectAllPathNameSet)>=batch_size or len(DefectAllPathNameSet)>=len(files):
                        #对当前批次的图像进行检测
                        g_index+=1
                        g_index_begin+=1
                        Print('Normal>>开始第{}批次检测!'.format(g_index))
                        log.AddLog('Normal>>开始第{}批次检测!'.format(g_index))
                        st = time.time()
                        print(DefectAllPathNameSet)
                        _, img_arr=classifier.GetImage(DefectAllPathNameSet)
                        if np.any(img_arr == None):
                            Print("Warning>>没有图像数据！")
                            log.AddLog("Warning>>没有图像数据！")
                            DeleteFile(DefectAllPathNameSet)
                            DefectAllPathNameSet = []
                            g_index_end += 1
                            continue
                        ##调用YOLO接口预测结果[[[],[]],[[],[]]]，返回的是每张图片的bbox ：shape(-1,6)的各种信息，想要查看的结果存放在path路径中，默认None，不保存结果
                        #！！！注意，所有bbox里面的信息顺序是cls, score, left, top, right, bottom，元素type = float32
                        every_img_boxes_list,class_name = yolo_Interface(yolo, DefectAllPathNameSet,path=r"steel_res_img/")#path = r"steel_res_img/"
                        print('NMS后BatchForEverybox的结果:',every_img_boxes_list)
                        if len(every_img_boxes_list) == 0:
                            Print("Remind>>没有缺陷信息！")
                            DeleteFile(DefectAllPathNameSet)
                            DefectAllPathNameSet =[]
                            g_index_end += 1
                            continue
                        Print('Normal>>第{}批次数据检测完成!'.format(g_index))
                        # 变量初始化
                        DefectList = []
                        for i in range(0, CameraNum):
                            DefectListOnlyCam = []
                            DefectList.append(DefectListOnlyCam)
                        num_imgs = len(DefectAllPathNameSet)
                        for i in range(num_imgs):  #遍历一个批次的所有数据
                            _,SteelNo,CamNo,ImgIndex, LeftPos,SteelCurrLen,Fx,Fy,_= DefectAllPathNameSet[i].split('_') #path
                            every_img_boxes = every_img_boxes_list[i]   #box_list [[],[],[]]
                            nameS = 0
                            for every_box_for_OneImg in every_img_boxes:
                                cls_id = int(every_box_for_OneImg[0])
                                score = every_box_for_OneImg[1]
                                ##原图坐标系
                                left = int(every_box_for_OneImg[2]) #x1
                                top = int(every_box_for_OneImg[3])  #y1
                                right = int(every_box_for_OneImg[4]) #x2
                                bottom = int(every_box_for_OneImg[5]) #y2
                                ##实际钢板坐标系
                                L = float(LeftPos)+float(left*float(Fx))
                                R = float(LeftPos)+float(right*float(Fx))
                                T = float(SteelCurrLen)+float(top*float(Fy))
                                B = float(SteelCurrLen)+float(bottom*float(Fy))
                                DefectNo, Cycle = 0, 0
                                className_for_Ebox =class_name[cls_id]
                                ExteralClassNo = classname_exteral[className_for_Ebox]
                                Grade = int(score * 100)
                                Area = abs(int(R)-int(L))*abs(int(B)-int(T))
                                AreaInImage = abs(int(right) - int(left)) * abs(int(bottom) - int(top)) #用于后处理缺陷设置
                                ImgData = None

                                if IsSave:
                                    if int(ExteralClassNo) == 0 and Grade >= 100 * 0.1:
                                        save_dir = r"E:\ArNTClassifiedSaveImageSet"
                                        if not os.path.exists(save_dir):
                                            os.mkdir(save_dir)
                                        sub_save_dir = os.path.join(save_dir, str(className_for_Ebox))
                                        if not os.path.exists(sub_save_dir):
                                            os.mkdir(sub_save_dir)
                                        imgName = str(nameS)+ str(DefectAllPathNameSet[i]).split('\\')[-1]
                                        imgPath = os.path.join(sub_save_dir, imgName)
                                        img = Image.open(DefectAllPathNameSet[i])
                                        cropped = img.crop((left, top, right, bottom))  # (left, top, right, bottom)
                                        cropped.save(imgPath)
                                        nameS += 1
                                    elif int(ExteralClassNo) != 0 and Grade >= 100 * 0.10:
                                        save_dir = r"E:\ArNTClassifiedSaveImageSet"
                                        if not os.path.exists(save_dir):
                                            os.mkdir(save_dir)
                                        sub_save_dir = os.path.join(save_dir, str(className_for_Ebox))
                                        if not os.path.exists(sub_save_dir):
                                            os.mkdir(sub_save_dir)
                                        imgName = str(nameS) + str(DefectAllPathNameSet[i]).split('\\')[-1]
                                        imgPath = os.path.join(sub_save_dir, imgName)
                                        img = Image.open(DefectAllPathNameSet[i])
                                        cropped = img.crop((left, top, right, bottom))  # (left, top, right, bottom)
                                        cropped.save(imgPath)
                                        nameS += 1
                                # 判断哪个相机是边部
                                # top_L_ini = 1
                                # top_L = 1
                                # top_R_ini = CameraNum//2
                                # top_R = CameraNum//2
                                # bot_L_ini = CameraNum//2 + 1
                                # bot_L = CameraNum//2 + 1
                                # bot_R_ini = CameraNum
                                # bot_R = CameraNum
                                ###----后续C++哪里改下代码，传出LeftEdge和RightEdge，在进行解封-----###
                                # if IsShieldEdge and int(ExteralClassNo) != 0:
                                #     # top
                                #     if int(CamNo) == top_L_ini:  # 1
                                #         # if int(CamNo) ==top_L_ini or int(CamNo) >=2 and float(LeftEdge) >= 70:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if 4096 - float(LeftEdge) <= 300:
                                #             top_L = top_L_ini + 1  # 2
                                #
                                #     if int(CamNo) >= top_L_ini + 1 and float(LeftEdge) >= EdgeNoDetectPixel:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if 4096 - float(LeftEdge) <= 300:
                                #             top_L = int(CamNo) + 1
                                #
                                #     if int(CamNo) == top_L:  # 相机1图片#  --2
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #
                                #     if int(CamNo) == top_R_ini:  # --4
                                #         if (float(Right) - float(Left)) / 2 + float(Left) > float(
                                #                 RightEdge) - ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if int(RightEdge) - int(LeftEdge) <= 300:
                                #             top_R = top_R_ini - 1
                                #     if int(CamNo) <= top_R_ini - 1 and 4096 - float(
                                #             RightEdge) >= EdgeNoDetectPixel:
                                #         if float(RightEdge) - (float(Right) - float(Left)) / 2 + float(
                                #                 Left) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if float(RightEdge) - int(LeftEdge) <= 300:
                                #             top_R = int(CamNo) - 1
                                #     if int(CamNo) == top_R:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) > float(
                                #                 RightEdge) - ShieldEdge:
                                #             ExteralClassNo = '0'
                                #
                                #     # bottom
                                #
                                #     if int(CamNo) == bot_L_ini:  # 1
                                #         # if int(CamNo) ==top_L_ini or int(CamNo) >=2 and float(LeftEdge) >= 70:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if 4096 - float(LeftEdge) <= 300:
                                #             bot_L = bot_L_ini + 1  # 2
                                #
                                #     if int(CamNo) >= bot_L_ini + 1 and float(LeftEdge) >= EdgeNoDetectPixel:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if 4096 - float(LeftEdge) <= 300:
                                #             bot_L = int(CamNo) + 1
                                #
                                #     if int(CamNo) == bot_L:  # 相机1图片#  --2
                                #         if (float(Right) - float(Left)) / 2 + float(Left) - float(
                                #                 LeftEdge) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #
                                #     if int(CamNo) == bot_R_ini:  # --4
                                #         if (float(Right) - float(Left)) / 2 + float(Left) > float(
                                #                 RightEdge) - ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if int(RightEdge) - int(LeftEdge) <= 300:
                                #             bot_R = bot_R_ini - 1
                                #     if int(CamNo) <= bot_R_ini - 1 and 4096 - float(
                                #             RightEdge) >= EdgeNoDetectPixel:
                                #         if float(RightEdge) - (float(Right) - float(Left)) / 2 + float(
                                #                 Left) < ShieldEdge:
                                #             ExteralClassNo = '0'
                                #         if float(RightEdge) - int(LeftEdge) <= 300:
                                #             bot_R = int(CamNo) - 1
                                #     if int(CamNo) == bot_R:
                                #         if (float(Right) - float(Left)) / 2 + float(Left) > float(
                                #                 RightEdge) - ShieldEdge:
                                #             ExteralClassNo = '0'

                                if len(class_name)==1 and IsUseClsArith:
                                    img_list = []
                                    img = Image.open(DefectAllPathNameSet[i])
                                    cropped = img.crop((left, top, right, bottom))  # (left, top, right, bottom)
                                    img_list.append(cropped)
                                    _,img_arr= classifier.GetMinImage(img_list)
                                    class_score, _, conf_score = classifier.PredictImages(img_arr)
                                    ClassifierClassNo = class_score
                                    InteralClassNo = tabel2[ClassifierClassNo]
                                    ExteralClassNo = tabel1[InteralClassNo]
                                    Grade = int(conf_score * 100)

                                if int(ExteralClassNo) != 0:
                                    # 写缺陷信息到列表
                                    DataItem = []
                                    DataItem.append(DefectNo)
                                    DataItem.append(SteelNo)
                                    DataItem.append(CamNo)
                                    DataItem.append(ImgIndex)
                                    DataItem.append(ExteralClassNo)
                                    DataItem.append(Grade)
                                    DataItem.append(left)
                                    DataItem.append(right)
                                    DataItem.append(top)
                                    DataItem.append(bottom)
                                    DataItem.append(L)
                                    DataItem.append(R)
                                    DataItem.append(T)
                                    DataItem.append(B)
                                    DataItem.append(Area)
                                    DataItem.append(Cycle)
                                    DataItem.append(ImgData)
                                    DefectList[int(CamNo) - 1].append(DataItem)
                                    # print(DataItem)
                                    # 将检测到的缺陷保存到sql数据库中

                        thread = []
                        for i in range(0, CameraNum + 1):
                            if i == 0:
                                t = threading.Thread(target=DeleteFile, args=(DefectAllPathNameSet,))
                                # t = threading.Thread(target=DeleteFile,args=([],))
                            else:
                                t = threading.Thread(target=WriteDatabase, args=(i, g_DefectDBIP, DefectList))
                            thread.append(t)
                        for i in range(0, CameraNum + 1):
                            thread[i].start()
                        for i in range(0, CameraNum + 1):
                            thread[i].join()

                        Print('Normal>>多线程写入缺陷成功!')
                        # DeleteFile(DefectAllPathNameSet)
                        Print('Normal>>完成批次{}共{}幅图像所用时间为{}秒!'.format(g_index, num_imgs, time.time() - st))
                        log.AddLog('Normal>>完成批次{}共{}幅图像所用时间为{}秒!'.format(g_index, num_imgs, time.time() - st))
                        g_index_end += 1
                        DefectAllPathNameSet = []
                else:
                    time.sleep(1)
                    Print("Warning>>数据类型不是BMP的文件存在！")
                    log.AddLog("Warning>>数据类型不是BMP的文件存在！")
            break

if __name__=="__main__":
    Main()
