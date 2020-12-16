import os
import shutil
import random


path=r"I:\ImageSampleSets\RZ_ImgSets_Total"


def Split_Validation_Train(path):
    total_path=os.path.join(path,"data")
    train_path=os.path.join(path,"train")
    validation_path=os.path.join(path,"validation")

    for sub_dir in os.listdir(total_path):
        total_sub_path=os.path.join(total_path,sub_dir)
        train_sub_path=os.path.join(train_path,sub_dir)
        validation_sub_path=os.path.join(validation_path,sub_dir)
        if True==os.path.exists(train_sub_path):
            shutil.rmtree(train_sub_path)
        if True==os.path.exists(validation_sub_path):
            shutil.rmtree(validation_sub_path)
        os.makedirs(train_sub_path)
        os.makedirs(validation_sub_path)
        for _,_,filenames in os.walk(total_sub_path):
            random.shuffle(filenames)
            file_n=len(filenames)
            if file_n<10:
                info="Error:<{}>文件夹数量小于10个，不符合要求，请检查...".format(sub_dir)
                input(info)
            for i in range(file_n):
                file_path=os.path.join(total_sub_path,filenames[i])
                if i<int(file_n/10):
                    validation_file_path=os.path.join(validation_sub_path,filenames[i])
                    shutil.copyfile(file_path,validation_file_path)
                else:
                    train_file_path=os.path.join(train_sub_path,filenames[i])
                    shutil.copyfile(file_path,train_file_path)
        info="Normal:<{}>文件夹图像已成功分离训练集和验证集！".format(sub_dir)
        print(info)
    
if __name__ == "__main__":
    Split_Validation_Train(path)
