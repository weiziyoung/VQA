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


def loss_figure():
    data = torch.load('loss.pth')
    count_loss_list = []
    base_loss_list = []
    for each in data:
        count_loss, base_loss = [_.data for _ in each]
        count_loss_list.append(float(count_loss))
        base_loss_list.append(float(base_loss))
    x = torch.linspace(0, 1000, 1000)
    plt.plot([float(each) for each in x], count_loss_list, label='count')
    plt.plot([float(each) for each in x], base_loss_list, label='base_line')
    plt.xlabel('Iteration(n)')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('$Loss\ curve(noise=0, object\_length=0.2, object=10)$')
    plt.show()



if __name__ == "__main__":
    loss_figure()