import os
import numpy as np


classes_path = 'F:\GQLianzhu\Predict\steel_model\steel_classes.txt'
#  cls, score, left, top, right, bottom
bboxes = [[5, 0.111609384, 2992, 425, 3488, 678], [1, 0.10897241, 2992, 425, 3488, 678],
        [1, 0.24340352, 3369, 933, 3447, 1014], [0, 0.11002868, 3369, 933, 3447, 1014],
        [0, 0.1288341, 3236, 502, 3465, 664], [1, 0.19032562, 3366, 937, 3446, 1024],
        [1, 0.31486464, 3246, 509, 3467, 679], [0, 0.1170744, 3366, 937, 3446, 1024]]

bboxes_arr = np.array(bboxes)
# print(bbox[1,:][1])
def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

cls_name = get_class(classes_path)
print(cls_name)


The_boxes_after_NMS =[]
for every_cls_index in range(len(cls_name)):
    #print(np.where(bbox[:,0] == every_cls_index)[0])
    #  cls, score, left, top, right, bottom
    the_boxes = bboxes_arr[np.where(bboxes_arr[:,0] == every_cls_index)[0], :]
    #print(the_boxes)
    # print(every_cls_index,the_boxes)
    x1 = the_boxes[:, 2]
    y1 = the_boxes[:, 3]
    x2 = the_boxes[:, 4]
    y2 = the_boxes[:, 5]
    scores = the_boxes[:, 1]
    #print(scores)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)
    #print(order)
    #The_boxes_after_NMS = []

    while order.size > 0:
        # 将当前置信度最大的框加入返回值列表中
        index = order[-1]
        The_boxes_after_NMS.append(the_boxes[index].tolist())

        # 获取当前置信度最大的候选框与其他任意候选框的相交面积
        x11 = np.maximum(x1[index], x1[order[:-1]])
        y11 = np.maximum(y1[index], y1[order[:-1]])
        x22 = np.minimum(x2[index], x2[order[:-1]])
        y22 = np.minimum(y2[index], y2[order[:-1]])
        w = np.maximum(0.0, x22 - x11 + 1)
        h = np.maximum(0.0, y22 - y11 + 1)
        intersection = w * h
        # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
        ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ious < 0.5)
        order = order[left]
print(The_boxes_after_NMS)
print(len(The_boxes_after_NMS))

# print(bboxes)
# print(len(bboxes))
# print(The_boxes_after_NMS)
# print(len(The_boxes_after_NMS))


def NMS(BBox_list,Class_name,threshold):
    bboxes_arr = np.array(BBox_list)
    The_boxes_after_NMS = []
    for every_cls_index in range(len(Class_name)):
        # print(np.where(bbox[:,0] == every_cls_index)[0])
        #  cls, score, left, top, right, bottom
        the_boxes = bboxes_arr[np.where(bboxes_arr[:, 0] == every_cls_index)[0], :]
        #print(the_boxes)
        # print(every_cls_index,the_boxes)
        x1 = the_boxes[:, 2]
        y1 = the_boxes[:, 3]
        x2 = the_boxes[:, 4]
        y2 = the_boxes[:, 5]
        scores = the_boxes[:, 1]
        #print(scores)
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = np.argsort(scores)
        #print(order)
        #The_boxes_after_NMS = []

        while order.size > 0:
            # 将当前置信度最大的框加入返回值列表中
            index = order[-1]
            The_boxes_after_NMS.append(the_boxes[index].tolist())

            # 获取当前置信度最大的候选框与其他任意候选框的相交面积
            x11 = np.maximum(x1[index], x1[order[:-1]])
            y11 = np.maximum(y1[index], y1[order[:-1]])
            x22 = np.minimum(x2[index], x2[order[:-1]])
            y22 = np.minimum(y2[index], y2[order[:-1]])
            w = np.maximum(0.0, x22 - x11 + 1)
            h = np.maximum(0.0, y22 - y11 + 1)
            intersection = w * h
            # 利用相交的面积和两个框自身的面积计算框的交并比, 将交并比大于阈值的框删除
            ious = intersection / (areas[index] + areas[order[:-1]] - intersection)
            left = np.where(ious < threshold)
            order = order[left]

    return The_boxes_after_NMS



print('----------分割线----------------')
list = NMS(bboxes,cls_name,0.5)
print(list)
print(len(list))
