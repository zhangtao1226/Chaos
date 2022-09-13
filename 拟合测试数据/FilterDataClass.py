# -*- coding: utf-8 -*-
# @Author  : zhangtao
# @File    : FilterDataClass.py
# @Desc    : 去噪点类
# @Time    : 2022/9/8 15:31
# @Software: PyCharm

import numpy as np
from math import e
import pandas as pd
from scipy import optimize, stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 显示中文
plt.rcParams['font.sans-serif'] = 'Songti Sc'
plt.rcParams['axes.unicode_minus'] = False

class filterDataClass:
    def __init__(self):
        pass

    def skewness(self, data, method):
        """
        偏度计算
        :param data: DataFrame 格式的数据源，包括：日期、监控项的值
        :param method: 噪点滤除方式：1、3-sigma；2、箱型图
        :return: DataFrame 格式的过滤后数据，
        """
        skew = stats.skew(data, bias=False)
        if skew > 1.27:
            print('高度正偏度， 需要做对数变换')
            edited_data = np.log(data)
            edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
            edited_data = e ** self.filter_data(edited_data, method, skew)
        elif skew > 0.65:
            print('中度正偏度，需进行开方变换')
            edited_data = np.sqrt(data)
            edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
            edited_data = self.filter_data(edited_data, method, skew) ** 2
        elif skew < -1.2:
            print('高度负偏度，需用最大值减去每个值再进行对数对数变换')
            edited_data = np.log(np.max(data) + 1 - data)
            edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
            edited_data = np.max(data) + 1 - (e ** self.filter_data(edited_data, method, skew))
        elif skew < -0.69:
            print('中度负偏度，需用最大值减去每个值再进行对数开方变换')
            edited_data = np.sqrt(np.max(data) + 1 - data)
            edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
            edited_data = np.max(data) + 1 - (self.filter_data(edited_data, method, skew) ** 2)
        else:
            print('正常')
            edited_data = data
            edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
        return skew, edited_data

    def filter_data(self, data, method=1, skew=None):
        """
        3-sigma 和 箱型图去噪点
        :param data: DataFrame 格式数据
        :param method: 1、3-sigma；2、箱型图
        :param skew: 偏度；为None时计算上下限进行筛选；小于-0.69时进行下筛选；大于0.65时进行向上筛选
        :return:
        """
        up_threshold = None
        down_threshold = None

        if method == 1:
            # 方式一：
            values = data['监控项的值']
            sigma_mean = np.mean(values)
            sigma_std = np.std(values)
            down_threshold = sigma_mean - 3 * sigma_std
            up_threshold = sigma_mean + 3 * sigma_std
            # 方式二：
            down_threshold = data['监控项的值'].quantile(0.03)
            up_threshold = data['监控项的值'].quantile(0.97)
        elif method == 2:
            Q3 = data['监控项的值'].quantile(0.75)
            Q1 = data['监控项的值'].quantile(0.25)
            up_threshold = Q3 + 1.5 * (Q3 - Q1)
            down_threshold = Q1 - 1.5 * (Q3 - Q1)

        # 根据上下限进行数据筛选
        if skew is None:
            df = data[(data['监控项的值'] > down_threshold) & (data['监控项的值'] < up_threshold)]
        elif skew > 0.65:
            df = data[data['监控项的值'] > down_threshold]
        elif skew < -0.69:
            df = data[data['监控项的值'] < up_threshold]
        return df

    def rolling_filter(self, kpi_data, window, method):
        """
        :param kpi_data: 进行移动计算的数据，dataframe格式，包括“监控项的值”和"日期"
        :param window: 用户输入的时间框大小（点的个数）
        :param method: 数据清洗类型（1：3-sigma；2：箱型图）
        :return: 筛选后的数据（dataframe格式，包含"日期"和"监控项的值"）
        """
        df = kpi_data.groupby(kpi_data.index // window).apply(lambda x: self.filter_data(x, type))
        return df.reset_index()[['日期', '监控项的值']]

    def changing_rate(self, kpi_data, window, fun_type):
        """
        :param kpi_data: 进行变化率计算的数据，dataframe格式，包括“日期”和“监控项的值”
        :param window: 用户输入的时间框大小（点的个数）
        :param fun_type: 变化率计算方式（ 1：线性拟合斜率；2：最大值和最小值差值）
        :return: 总体变化率(float) 和 时间窗内的变化率(numpy.ndarray)，其单位为每分钟的变化率
        """
        k = None
        values = kpi_data['监控项的值']
        times = pd.to_datetime(kpi_data['日期'])
        value_df = pd.DataFrame(values.values.tolist(), index=times)
        if fun_type == 1:
            expanding_seconds = (times.iloc[-1] - times.iloc[0]).days
            expanding_minutes = expanding_seconds / 1
            # 拟合斜率-总体变化率
            k, b = optimize.curve_fit(self.fun_1, np.arange(
                kpi_data.shape[0]), values)[0]
            value_df = value_df.rolling(window, min_periods=window).apply(
                lambda x: self.curve_fit_fun(x) / (expanding_minutes / len(values)))

        elif fun_type == 2:
            expanding_seconds = (times[np.argmax(values)] - times[np.argmin(values)]).days
            expanding_minutes = expanding_seconds / 1
            # 最大值和最小值差值-总体变化率
            k = (max(values) - min(values)) / (np.argmax(values) - np.argmin(values))
            value_df = value_df.rolling(window, min_periods=window).apply(
                lambda x: self.min_max_fit_fun(x))
        value_df['日期'] = value_df.index
        value_df['监控项的值'] = value_df[0]
        return k, value_df[['日期', '监控项的值']]

    def curve_fit_fun(self, values):
        """
        变化率计算方式：拟合函数斜率
        :param values:
        :return:
        """
        k, b = optimize.curve_fit(self.fun_1, np.arange(len(values)), values)[0]
        return k

    def min_max_fit_fun(self, values):
        """
        变化率计算方式：最大最小值的差值
        :param values:
        :return:
        """
        times = pd.to_datetime(values.index)
        expanding_seconds = abs((times[np.argmax(values)] - times[np.argmin(values)]).days)
        expanding_minutes = expanding_seconds / 1
        k = (max(values) - min(values))
        return k / (expanding_minutes / (np.argmax(values) - np.argmin(values)))

    def fitting_fun(self, data1, data2, fit_type):
        """
        :param data1: 参与拟合数据1，dataframe格式
        :param data2: 参与拟合数据2，dataframe格式
        :param fit_type: 拟合方式：1：一次线性拟合；2：二次拟合方式
        :return: 拟合后数据y_pre，拟合效果R2
        """
        # 对两组数据进行拟合回归【f_1：线性拟合， f_2：二次非线性拟合】
        if fit_type == 1:
            A1, B1 = optimize.curve_fit(self.fun_1, data1, data2)[0]
            print('拟合系数：', A1, B1)
            y_pre = A1 * np.array(data1) + B1
            print('拟合公式: y = ' + str(round(A1, 3)) + ' * x + ' + str(round(B1, 3)))
        elif fit_type == 2:
            A1, B1, C1 = optimize.curve_fit(self.fun_2, data1, data2)[0]
            y_pre = A1 * np.array(data1) * np.array(data1) + B1 * np.array(data1) + C1
            print('拟合公式: y = ' + str(round(A1, 3)) + ' * x^2 + ' +
                  str(round(B1, 3)) + ' * x + ' + str(round(C1, 3)))
        R2 = r2_score(data2.tolist(), y_pre)
        print('拟合效果R2： ', R2)
        return R2, y_pre

    def alignment(self, data1, data2, method=1):
        """
        :param data1: 参与补点数据1，dataframe格式，包括时间和数据
        :param data2: 参与补点数据2，dataframe格式，包括时间和数据
        :param method: 补点类型 1：以前一个有值数据进行补点；2：等差数列进行补点
        :return: 补点后数据，data1和data2
        """
        data = pd.merge(data1, data2, on=['日期'], how='outer')
        data = data.sort_values(by='日期')
        if method == 1:
            # 以前一个数据为基准进行补点

            for cate_id in data.columns:
                if cate_id != '日期':
                    for i in range(data.shape[0]):
                        try:
                            if np.isnan(data[cate_id].iloc[i]):
                                data[cate_id].iloc[i] = data[cate_id].iloc[i - 1]
                        except:
                            pass

            for cate_id in data.columns:
                if cate_id != '日期':
                    for i in range(data.shape[0]):
                        try:
                            if np.isnan(data[cate_id].iloc[data.shape[0] - i]):
                                data[cate_id].iloc[data.shape[0] - i] = data[cate_id].iloc[data.shape[0] - i - 1]
                        except:
                            pass
        elif method == 2:
            # 根据空白值前后值进行分阶段给值，等差数列方式进行补点
            pass
        return data

    def fun_1(self, x, A, B):
        """
        一元线性回归函数
        :param x:
        :param A:
        :param B:
        :return:
        """
        return A * x + B

    def fun_2(self, x, A, B, C):
        """
        二次拟合
        :param x:
        :param A:
        :param B:
        :param C:
        :return:
        """
        return A * x * x + B * x + C

    def filter_data(self, data):
        """

        :param data:
        :return:
        """


if __name__ == '__main__':
    # df_1 = pd.read_excel(r'转速实际值2.xlsx')
    # df_2 = pd.read_excel(r'速度有效值2.xlsx')
    # df_1['日期'] = pd.to_datetime(df_1['日期'])
    # df_2['日期'] = pd.to_datetime(df_2['日期'])

    df = pd.read_excel('转速实际值与速度有效值补点后.xlsx')

    # 统计信息描述
    # print('转速实际值', df_1.describe())
    # print('速度有效值', df_2.describe())
    # df = df[['日期', '监控项的值']]

    filter_data = filterDataClass()
    # 补点
    # new_df = filter_data.alignment(df_1, df_2)
    # new_df.to_excel('转速实际值与速度有效值(时间对齐).xlsx', index=None)
    # print(new_df)
    # exit()
    # 差分
    # new_df = np.diff(df['监控项的值'])
    # new_df = list(new_df)

    # 箱形图
    # new_df = filter_data.filter_data(df, method=2)



    # print(new_df)
    # exit()

    # 去噪点, 去除阈值小于857.378000 下 25%
    new_df = df[df['监控项的值_x'] > 857.378]

    # 移动窗口，取最大值
    # size = 15
    # new_df = new_df.groupby(new_df.index // size).max()


    x = new_df['监控项的值_x']
    y = new_df['监控项的值_y']
    # 拟合
    R2, y_pre = filter_data.fitting_fun(x, y, 2)



    # print(new_df)


    # exit()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)

    # 拟合
    ax.set_ylabel('速度有效值')
    ax.set_xlabel('转速实际值')
    ax.plot(x, y, '.')
    ax.plot(x, y_pre, '*', color='r')
    # ax.plot(x, y_pre, color='r')

    #
    ax.legend(['转速实际值', '速度有效值'])
    plt.title('转速实际值与速度有效值(%s)'% R2)
    plt.grid()
    plt.show()

    exit()

    ax.set_ylabel('转速实际值')
    ax.set_xlabel('日期')
    # 原始曲线
    lines1 = ax.plot(new_df['日期'], new_df['监控项的值_x'])
    #
    ax1 = ax.twinx()
    lines2 = ax1.plot(new_df['日期'], new_df['监控项的值_y'], color='r')
    ax1.set_ylabel('速度有效值')
    lines = lines1 + lines2
    #
    ax.legend(lines, ['转速实际值', '速度有效值'], loc='upper left')
    plt.title('转速实际值-速度有效值(最大值)')




    plt.grid()
    plt.show()