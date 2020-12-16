from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def gen_data(path,gen_dir,exp_num=10):
    datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    if False==os.path.exists(gen_dir):
        os.mkdir(gen_dir)

    for sub_dir in os.listdir(path):
        category = []
        sub_dir_path = os.path.join(path,sub_dir)
        print("开始生成的类别文件：",sub_dir)
        n = 0
        for _,_,filenames in os.walk(sub_dir_path):
            for filename in filenames:
                image_path = os.path.join(sub_dir_path,filename)
                img = load_img(image_path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                index = 0
                save_dir=gen_dir+'/'+sub_dir
                if False==os.path.exists(save_dir):
                    os.mkdir(save_dir)
                for batch in datagen.flow(x, batch_size=1, save_to_dir=save_dir, save_prefix="exp", save_format='bmp'):
                    index += 1
                    n += 1
                    if index >= exp_num:
                        break
                print("正在生成图片数量：",n)

if __name__=="__main__":
    path = r"I:\ImageSampleSets\RZ_ImgSets_Total\data"
    exp_num=10
    gen_dir="datagen"
    gen_data(path,gen_dir,exp_num)
