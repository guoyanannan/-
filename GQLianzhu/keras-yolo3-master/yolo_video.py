import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import time
import cv2
# import  tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)

import glob,os
from skimage import io
from matplotlib import pyplot as plt
import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

#批量检测图片，每检测一张进行展示且保存到指定文件夹

# input_dir = './test/*.jpg'
# output_dir = './TestResult/'



def detect_img(yolo):
    path = "./test/*.jpg"  #待检测图片的位置
    # 存储检测结果的目录
    outputdir = "./TestResult/"
    t1 = time.time()
    for jpgfile in glob.glob(path):
        imagename = os.path.basename(jpgfile)
        img = Image.open(jpgfile)
        img = yolo.detect_image(img,jpgfile)
        image_save_path = outputdir+'result_'+imagename
        print('detect result save to....:' +image_save_path)
        img.save(image_save_path)
        #展示图像'
        img = np.array(img)
        io.imshow(img)
        plt.show()
    time_sum = time.time()-t1
    yolo.detect_time(time_sum)
    # file.write('time sum: '+str(time_sum)+'s')
    print('time sum:',time_sum)
    #file.close()
    yolo.close_session()

# def detect_img(yolo):
#     if not os.path.exists(output_dir):
#         os.mkdir(output_dir)
#     filename_list = glob.glob(input_dir)
#     for filename in filename_list:
#         img = Image.open(filename)
#         print('------------------------------------')
#         img = yolo.detect_image(img)
#         print('------------------------------------')
#
#         img.save(os.path.join(output_dir, os.path.basename(filename)))
#         print('------------------------------------')
#         # img = np.array(img)
#         # io.imshow(img)
#         # plt.show()
#     yolo.close_session()


	

'''
#定义检测图片函数
def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()
'''
FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=True, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:#file.close()
        print("Must specify at least video_input_path.  See usage with --help.")
