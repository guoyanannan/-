from PIL import Image
import os
from tqdm import tqdm
import cv2

def split_img(img_path,win_size,stride):
    import cv2
    sub_imgs = []
    img = cv2.imread(img_path)
    img2 = Image.open(img_path)
    # print(img2.size,len(img2.split()))
    chanel = cv2.split(img)
    print(len(chanel))
    print(img.shape)
    sub_imgs.append(img)
    h, w, _ = img.shape
    #win_size = 1024 ,stride =768
    for r in range(0, (h - win_size) + 1, stride):  # H方向进行切分
        for c in range(0, (w - win_size) + 1, stride):  # W方向进行切分
            tmp = img[r: r + win_size, c: c + win_size]
            sub_imgs.append(tmp)
    return sub_imgs


path = '../DataOrigin/HuaHen'

for No, name in tqdm(enumerate(os.listdir(path)[:2])):
    if name.split('.')[-1] == 'bmp':
        # print(No, name, name.rstrip('.bmp'))
        file_path = os.path.join(path,name)
        img_set=split_img(file_path,1024,768)
        print(img_set[0].shape,img_set[1].shape,img_set[2].shape,img_set[3].shape,img_set[4].shape,img_set[5].shape)