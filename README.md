# Keras_yolov3工程化，对图像进行了切分，坐标重构
#GQlianzhu/DataOrigin/ NMSTest.py  ----离线测试NMS
#GQlianzhu/DataOrigin/ SplitImg_and_RewriteXML.py  ----切分图像和坐标转换
#GQlianzhu/DataOrigin/SplitImg_test.py  ----离线测试切图
#GQlianzhu/DataOrigin/ TestBBox.py  ----测试图像切割后，转换后的坐标是否正确
#GQlianzhu/DataOrigin/ TestBBoxXML.py  ----测试图像切割后，转换后的坐标是否正确，单个类别
#其它文件夹存以缺陷类别名称为定义，存放数据和标注文件

##keras_yolo3_master 就什么可说的了，大神开源的代码，在此基础上略微修改增加mAP的计算


###Predict是推理接口，steel_model下需要--检测的类别、锚框文件--内外编号转换文件--分类索引和内部编号的ini文件--以及分类模型文件和检测模型文件
#把对应的参数里面的模型、以及训练数据放置于百度网盘###


