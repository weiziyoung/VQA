# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 4:54 AM
# @Author  : weiziyang
# @FileName: new.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

cm = plt.cm.coolwarm
axis = plt.gca()
rect = patches.Rectangle((0.5, 0.5), width=0.5, height=0.5, linewidth=1, fill=True, alpha=0.7, color=cm(float(1)))
axis.add_patch(rect)
plt.show()