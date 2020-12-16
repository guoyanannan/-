import os
import numpy as np


def SteelNMS(BBox_list,Class_name,threshold):
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






