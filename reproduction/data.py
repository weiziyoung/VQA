# -*- coding: utf-8 -*-
# @Time    : 2019/5/12 8:50 PM
# @Author  : weiziyang
# @FileName: data.py
# @Software: PyCharm

import torch.utils.data as data
import visualization
import torch
import random


class ToyTask(data.Dataset):
    """
    This class is designed to yield fake dataset, so as to test the performance of our counting model.
    """
    def __init__(self, object_num=10, length=0.2):
        super().__init__()
        self.object_num = object_num
        if length:
            self.object_length = length
        else:
            self.object_length = 0.05 + torch.rand(object_num) * 0.3

    def __getitem__(self, item):
        # generate left-upper coordinate,prevent the object  from overflowing the screen
        left_upper_cord = torch.rand(2, self.object_num)*(1-self.object_length)
        boxes = torch.cat([left_upper_cord,  left_upper_cord + self.object_length])
        true_num = random.randint(1, self.object_num)
        true_boxes = boxes[:, :true_num]
        iou = self.iou(boxes, true_boxes)
        # The larger weight, the color darker
        weights = iou.max(dim=1)[0]
        return weights, boxes, true_num

    def iou(self, a, b):
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(1).expand_as(inter)
        area_b = self.area(b).unsqueeze(0).expand_as(inter)
        return inter / (area_a + area_b - inter)

    def area(self, box):
        x = (box[2, :] - box[0, :]).clamp(min=0)
        y = (box[3, :] - box[1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (2, a.size(1), b.size(1))
        min_point = torch.max(
            a[:2, :].unsqueeze(dim=2).expand(*size),
            b[:2, :].unsqueeze(dim=1).expand(*size),
        )
        max_point = torch.min(
            a[2:, :].unsqueeze(dim=2).expand(*size),
            b[2:, :].unsqueeze(dim=1).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[0, :, :] * inter[1, :, :]
        return area


if __name__ == "__main__":
    a = ToyTask(object_num=10)
    print(a['1'])