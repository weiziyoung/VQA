# -*- coding: utf-8 -*-
# @Time    : 2019/5/16 11:57 AM
# @Author  : weiziyang
# @FileName: myplot.py
# @Software: PyCharm
import torch
import numpy as np
import matplotlib.pyplot as plt


def real_situation():
    data = torch.load('real_situation.pth')
    count_acc = []
    base_acc = []
    noise = []
    for each in data:
        count, base = each['accs']
        count_acc.append(count)
        base_acc.append(base)
        noise.append(each['noise'])
    plt.figure(figsize=(5, 5))
    plt.plot(noise, count_acc, label='counting')
    plt.plot(noise, base_acc, label='base')
    plt.xlabel('noise')
    plt.ylabel('accuracy')
    plt.title('Accuracy of Random Object length')
    plt.legend()
    plt.show()



if __name__ == "__main__":
    real_situation()