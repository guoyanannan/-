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
        #print(class_convert_tabel)
        return class_convert_tabel,class_convert_tabel_1
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
