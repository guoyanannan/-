################################################################################
##########################         BKVision        #############################
##########################       wu kun peng       #############################
##########################       version 1.0       #############################
################################################################################

from . import inception_v3
from . import resnet50
from . import xception
from . import densenet
from . import nasnet
from . import mobilenet_v2
from . import inception_resnet_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.callbacks import *
from PIL import Image
import numpy as np
import keras
import scipy
import h5py
import os
import random
from optparse import OptionParser
import keras.backend.tensorflow_backend as KTF
from keras.models import model_from_json,load_model
import configparser
from xml.etree import ElementTree as ET
from TOOLS.file_oper import read_class_table_name,read_config_file,write_config_file

###命令参数
##parser = OptionParser()
##parser.add_option("--img_path", dest="img_path", help="Path to training data",default=r"E:\在线检测系统分类训练文件夹\本机训练\data")
##parser.add_option("--img_size", type="int", dest="img_size", help="Size of Img.", default=192)
##parser.add_option("--network", dest="network", help="Base network to use.", default='InceptionResNetV2')
##parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=10)
##parser.add_option("--batch_size", type="int", dest="batch_size", help="Number of batch_size.", default=2)
##parser.add_option("--output_path", dest="output_path", help="Output path for models.", default='./model_steel.h5')
##(options, args) = parser.parse_args()
##
###配置参数
##img_width, img_height = options.img_size,options.img_size,#192, 192
##train_data_dir = options.img_path+'/train'
##validation_data_dir = options.img_path+'/validation'
##test_data_dir = options.img_path+'/test'
##epochs = options.num_epochs
##batch_size = options.batch_size
##
##
##if False==os.path.exists(train_data_dir):
##    input("@@Error：目录下没有train文件夹，训练已停止，请修改！！！")
##if False==os.path.exists(validation_data_dir):
##    input("@@Error：目录下没有validation文件夹，训练已停止，请修改！！！")
##if epochs<=0:
##    input("@@Error：epochs必须大于0，训练已停止，请修改！！！")
##if batch_size<=0:
##    input("@@Error：batch_size必须大于0，训练已停止，请修改！！！")
##if options.network=="Inception_V3" and img_width < 139:
##    input("@@Error：Inception_V3模型要求尺寸必须大于等于139，训练已停止，请修改！！！")
##elif options.network=="ResNet50" and img_width < 197:
##    input("@@Error：ResNet50模型要求尺寸必须大于等于197，训练已停止，请修改！！！")

#读图像函数
def read_all_image_data(dir_name):
    samples_arr = []
    categorys_arr = []
    labels=[]
    classnum=0
    path = os.path.realpath(dir_name)
    for sub_dir in os.listdir(path):
        category = []
        sub_dir_path = os.path.join(path,sub_dir)
        for _,_,filenames in os.walk(sub_dir_path):
            num=0
            for filename in filenames:
                image_path = os.path.join(sub_dir_path,filename)
                im = Image.open(image_path)
                im = im.convert("RGB")
                im = im.resize((img_width,img_height))
                im = np.asarray(im,np.float32)
                im = im/255.0
                #im = im[...,np.newaxis]
                category.append(im)
                samples_arr.append(im)
                labels.append(classnum)
                num+=1
        category = np.stack(category,axis=0)
        categorys_arr.append({"x":category,"y":sub_dir})
        print(classnum,sub_dir,num)
        classnum+=1
        
    data_arr = np.stack(samples_arr,axis=0).astype(np.float32)
    labels_arr = np.stack(labels,axis=0).astype(np.float32)
    return data_arr,labels_arr
#读数量函数
def read_all_image_num(dir_name):
    ImageNum=0
    ClassNum=0
    path = os.path.realpath(dir_name)
    for sub_dir in os.listdir(path):
        ClassNum+=1
        category = []
        sub_dir_path = os.path.join(path,sub_dir)
        for _,_,filenames in os.walk(sub_dir_path):
            for filename in filenames:
                ImageNum+=1
    return ImageNum,ClassNum
#得到目录的类别名
def get_class_label_from_dir(dir_name):
    path = os.path.realpath(dir_name)
    category = []
    every_imgnum=[]
    for sub_dir in os.listdir(path):
        category.append(sub_dir)
        sub_dir_path = os.path.join(path,sub_dir)
        CurrImgNum=0
        for _,_,filenames in os.walk(sub_dir_path):
            CurrImgNum=len(filenames)
        every_imgnum.append(CurrImgNum)
    #print(category)
    return category,every_imgnum


#损失函数    
class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.losses=[]
        #self.acc=[]
    def on_batch_end(self,batch,logs={}):
        self.losses.append(logs.get('loss'))
        #self.acc.append(logs.get('acc'))

##训练函数
def train(sample_path,class_table_path,epochs,batch_size,img_size,network,bIsTraining=True):

    img_width, img_height = img_size,img_size,#192, 192
    train_data_dir = sample_path+'/train'
    validation_data_dir = sample_path+'/validation'
    
    if False==os.path.exists(train_data_dir):
        input("@@Error：目录下没有train文件夹，训练已停止，请修改！！！")
    if False==os.path.exists(validation_data_dir):
        input("@@Error：目录下没有validation文件夹，训练已停止，请修改！！！")
    if epochs<=0:
        input("@@Error：epochs必须大于0，训练已停止，请修改！！！")
    if batch_size<=0:
        input("@@Error：batch_size必须大于0，训练已停止，请修改！！！")

  
    checkpointer=ModelCheckpoint(filepath="model--{epoch:02d}--{acc:.6f}--{val_acc:.6f}.h5",verbose=1)
    history=LossHistory()

    train_data_num,train_class_num=read_all_image_num(train_data_dir)
    validation_data_num,validation_class_num=read_all_image_num(validation_data_dir)
    #test_data_num,test_class_num=read_all_image_num(test_data_dir)

    print("训练集图像数量：",train_data_num , "----训练集文件夹数量：",train_class_num)
    print("验证集图像数量：",validation_data_num , "----验证集文件夹数量：",validation_class_num)
    #print("测试集图像数量：",test_data_num , "----测试集文件夹数量：",test_class_num)

    if (train_class_num != validation_class_num):
        input("@@Warning：train或validation目录下的文件夹数量与指定类别不一致，训练已暂停，可点击回车继续！！！")
    num_classes=train_class_num

    #得到训练集的类别名
    train_class_names,train_every_imgnum=get_class_label_from_dir(train_data_dir)
    print(train_class_names)
    weights=[]
    for i in range(len(train_every_imgnum)):
        weight=float(train_every_imgnum[i])/train_data_num
        weights.append(weight)
    print(weights)
    validation_class_names,validation_every_imgnum=get_class_label_from_dir(validation_data_dir)
    print(validation_class_names)
    if train_class_names != validation_class_names:
        input("@@Error：train或validation目录下的文件夹类别名称不一致，训练已停止，请修改！！！")
    #读类别表
    class_convert_table,_ =read_class_table_name(class_table_path)
    print(class_convert_table)
    #训练类别名称转为内部类别编号
    train_class_interalno=[]
    for train_class_name in train_class_names:
        if True==(train_class_name in class_convert_table): 
            train_class_interalno.append(int(class_convert_table[train_class_name]))
        else:
            error="@@Error：<%s>类别在ArNTClassTable.xml未找到，训练已停止，请修改！！！" % train_class_name
            input(error)
    print(train_class_interalno)
    #写配置文件     
    write_config_file(train_class_num,train_class_interalno,img_size,network)

    architecture_path=r"steel_model_architecture.json"
    weights_path=r"steel_model_weights.h5"
    model_path=r"steel_model.h5"

    if bIsTraining == True:
        generator = ImageDataGenerator(rescale=1/255.0)
        train_generator = generator.flow_from_directory(train_data_dir,
                                                        color_mode = "rgb",
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        classes=train_class_names,
                                                        class_mode='categorical')
        validation_generator = generator.flow_from_directory(validation_data_dir,
                                                    color_mode = "rgb",
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    classes=validation_class_names,
                                                    class_mode='categorical')

        if network=="Inception_V3":
            base_model=inception_v3.InceptionV3(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="ResNet50":
            base_model=resnet50.ResNet50(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="Xception":
            base_model=xception.Xception(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="DenseNet":
            base_model=densenet.DenseNet201(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="NasNet_Large":
            base_model=nasnet.NASNetLarge(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="NasNet_Mobile":
            base_model=nasnet.NASNetMobile(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="MobileNetV2":
            base_model=mobilenet_v2.MobileNetV2(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
        elif network=="InceptionResNetV2":
            base_model=inception_resnet_v2.InceptionResNetV2(weights='imagenet',include_top=False,input_shape=[img_height,img_width,3])
            x = base_model.output
            for i,layer in enumerate(base_model.layers):
                print(i,layer.name)
        else:
            if False==os.path.exists("BaseModel.h5"):
                input("@@Error：未指定网络模型，或BaseModel.h模型文件不存在，训练已停止，请修改！！！")
            base_model=load_model("BaseModel.h5")
            num=0
            for i,layer in enumerate(base_model.layers):
                num+=1
                print(i,layer.name)
            x = base_model.layers[num-6].output
        
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.5)(x)
        #x = Dense(256, activation='relu')(x)
        predictions = Dense(train_class_num, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        if network=="Inception_V3":
            for layer in model.layers[:172]:#172
                layer.trainable = False
            for layer in model.layers[172:]:
                layer.trainable = True
        elif network=="ResNet50":
            for layer in model.layers[:120]:
                layer.trainable = False
            for layer in model.layers[120:]:
                layer.trainable = True
        elif network=="Xception":
            for layer in model.layers[:85]:
                layer.trainable = False
            for layer in model.layers[85:]:
                layer.trainable = True
        elif network=="DenseNet":
            for layer in model.layers[:480]:
                layer.trainable = False
            for layer in model.layers[480:]:
                layer.trainable = True
        elif network=="NasNet_Large":
            for layer in model.layers[:600]:
                layer.trainable = False
            for layer in model.layers[600:]:
                layer.trainable = True
        elif network=="NasNet_Mobile":
            for layer in model.layers[:496]:
                layer.trainable = False
            for layer in model.layers[496:]:
                layer.trainable = True
        elif network=="MobileNetV2":
            for layer in model.layers[:100]:
                layer.trainable = False
            for layer in model.layers[100:]:
                layer.trainable = True
        elif network=="InceptionResNetV2":
            for layer in model.layers[:440]:
                layer.trainable = False
            for layer in model.layers[440:]:
                layer.trainable = True
        else:
            for layer in model.layers[:172]:
                layer.trainable = False
            for layer in model.layers[172:]:
                layer.trainable = True  
             
        model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=["accuracy"])

        for i,layer in enumerate(model.layers):
            print(i,layer.name)
        print(">>开始训练：训练过程时间较长，请耐心等待！")
##        model.fit_generator(train_generator,
##                            samples_per_epoch=train_data_num,
##                            verbose=1,
##                            validation_steps=1,
##                            validation_data=validation_generator,
##                            nb_epoch=epochs,
##                            callbacks=[checkpointer,history])
        model.fit_generator(train_generator,
                            steps_per_epoch=train_data_num/batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=validation_generator,
                            validation_steps=validation_data_num/batch_size,
                            class_weight=weights,#None,weights,
                            shuffle=True,
                            callbacks=[checkpointer,history])        
        json_string = model.to_json() 
        open(architecture_path,'w').write(json_string) 
        model.save_weights(weights_path)
        model.save(model_path)
        input(">>：训练完成，可寻找最佳模型进行更换！")
    else :
        if test_data_num > 0:
            print("测试集详细信息：")
            TestImg,TestLabel = read_all_image_data(test_data_dir)
            print("开始载入模型")
            model=model_from_json(open(architecture_path).read())
            model.load_weights(weights_path)
            model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy',metrics=["accuracy"])
            print("模型载入成功")
            class_scores=model.predict(TestImg,batch_size=TestImg.shape[0])
            preds_result = []
            for class_score in class_scores:
                pre_class=0
                max_score=0.0
                for n in range(num_classes):
                    if class_score[n]>max_score:
                        max_score=class_score[n]
                        pre_class=n
                preds_result.append(pre_class)
            print(preds_result)
            print(TestLabel)
            corr=0
            error=0
            for i in range(test_data_num):
                if int(preds_result[i])==int(TestLabel[i]):            
                    corr+=1
                else:
                    error+=1
            print("测试集准确率：",float(corr)/test_data_num)








