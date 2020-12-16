import numpy as np
from PIL import Image
import os
import cv2
import pprint
import pickle

def cut_img(img_dir_list):
    index=0
    for item in os.listdir(img_dir_list):
        img_path=os.path.join(img_dir_list,item)
        print(img_path)
        gray = cv2.imread(img_path)
        #gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        width,height,_=gray.shape
        gray_cut=gray[20:width-20,20:height-20]
        
        #cv2.imshow("dst",img)
        cv2.imwrite(str(index)+'.bmp',gray_cut)
        index=index+1

