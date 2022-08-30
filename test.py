# -*- coding: utf-8 -*-
# @Author  : zhangtao
# @File    : test.py
# @Desc    :
# @Time    : 2022/8/29 18:18
# @Software: PyCharm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file_0 = '0偏差.xls'
file_1 = '1偏差.xls'
file_2 = '2偏差.xls'

data0 = pd.read_excel(file_0)
data1 = pd.read_excel(file_1)
data2 = pd.read_excel(file_2)

plt.plot(data0)
plt.plot(data1)
plt.plot(data2)
plt.show()
