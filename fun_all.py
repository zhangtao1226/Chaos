import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from scipy import stats
from math import e
plt.rcParams['font.sans-serif'] = 'Songti Sc'
plt.rcParams['axes.unicode_minus'] = False

'''
    数据清洗功能列表：
        1、数据滤除功能（3sigma或箱型图），对全体数据操作；filter_data
        2、偏度计算+数据滤除（根据偏度大小选择滤除类型）：skewness
        3、移动噪点去除（3-sigma或箱型图），根据用于输入的窗口大小进行滑动计算：rolling_filter
        4、变化率功能，根据用于输入的窗口大小进行滑动计算：changing_rate
        5、移动平均功能，根据用于输入的窗口大小进行滑动计算：moving_average
        6、统计信息功能，根据输入数据计算不同时间单位的均值和极值：Statistics
'''


# 偏度计算+数据滤除（根据偏度大小选择滤除类型）
def skewness(data, type):
    '''
    :param data: 进行偏度计算和噪点滤除的数据，dataframe格式，包括"日期"和"监控项的值"
    :param type: 噪点滤除方式（1：3-sigma；2：箱型图）
    :return edited_data： 进行滤除后的数据，dataframe格式，包括"日期"和"上下限"
    '''
    skew = stats.skew(data, bias=False)
    # 计算偏度
    '''
                skew	kurtosis
    中度正偏度	0.656308	0.584120
    高度正偏度	1.271249	2.405999
    中度负偏度	-0.690244	0.790534
    高度负偏度	-1.201891	2.086863
    '''
    # skew = skewness(data['监控项的值'])
    if skew > 1.27:
        print('高度正偏度， 需要做对数变换')
        edited_data = np.log(data)
        edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
        edited_data = e ** filter_data(edited_data, type, skew)
    elif skew > 0.65:
        print('中度正偏度，需进行开方变换')
        edited_data = np.sqrt(data)
        edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
        edited_data = filter_data(edited_data, type, skew) ** 2
    elif skew < -1.2:
        print('高度负偏度，需用最大值减去每个值再进行对数对数变换')
        edited_data = np.log(np.max(data)+1 - data)
        edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
        edited_data = np.max(data)+1 - (e ** filter_data(edited_data, type, skew))
    elif skew < -0.69:
        print('中度负偏度，需用最大值减去每个值再进行对数开方变换')
        edited_data = np.sqrt(np.max(data)+1 - data)
        edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
        edited_data = np.max(data)+1 - (filter_data(edited_data, type, skew) ** 2)
    else:
        print('正常')
        edited_data = data
        edited_data = pd.DataFrame(edited_data, columns=['监控项的值'])
    return skew, edited_data


# 数据滤除功能（3sigma或箱型图），对全体数据操作
def filter_data(data, type, skew=None):
    '''
    :param data: 进行噪点滤除的数据，dataframe格式，包括“监控项的值”和"日期"
    :param type: 噪点滤除方式（1：3-sigma；2：箱型图）
    :param skew: 偏度值，为None时计算上下限进行筛选；小于-0.69时进行下筛选；大于0.65时进行向上筛选；
    :return df: 筛选后的数据，dataframe格式，包括"日期"和"监控项的值"
    '''
    if type == 1:
        # 方式一：
        values = data['监控项的值']
        sigma_mean = np.mean(values)
        sigma_std = np.std(values)
        sigma_down_threshold = sigma_mean - 3 * sigma_std
        sigma_up_threshold = sigma_mean + 3 * sigma_std
        # 方式二：
        sigma_down_threshold = data['监控项的值'].quantile(0.03)
        sigma_up_threshold = data['监控项的值'].quantile(0.97)
    elif type == 2:
        Q3 = data['监控项的值'].quantile(0.75)
        Q1 = data['监控项的值'].quantile(0.25)
        sigma_up_threshold = Q3 + 1.5 * (Q3 -Q1)
        sigma_down_threshold = Q1 - 1.5 * (Q3 -Q1)
    # 根据上下限进行数据筛选
    if skew == None:
        df = data[(data['监控项的值'] > sigma_down_threshold) & (data['监控项的值'] < sigma_up_threshold)]
    elif skew > 0.65:
        df = data[data['监控项的值'] > sigma_down_threshold]
    elif skew < -0.69:
        df = data[data['监控项的值'] < sigma_up_threshold]
    return df


# 移动噪点去除（3-sigma或箱型图），根据用于输入的窗口大小进行滑动计算
def rolling_filter(kpi_data, window, type):
    '''
    :param kpi_data: 进行移动计算的数据，dataframe格式，包括“监控项的值”和"日期"
    :param window: 用户输入的时间框大小（点的个数）
    :param type: 数据清洗类型（1：3-sigma；2：箱型图）
    :return: 筛选后的数据（dataframe格式，包含"日期"和"监控项的值"）
    '''
    df = kpi_data.groupby(kpi_data.index // window).\
        apply(lambda x: filter_data(x, type))
    return df.reset_index()[['日期', '监控项的值']]


# 变化率功能，根据用于输入的窗口大小进行滑动计算
def changing_rate(kpi_data, window, fun_type):
    '''
    :param kpi_data: 进行变化率计算的数据，dataframe格式，包括“日期”和“监控项的值”
    :param window: 用户输入的时间框大小（点的个数）
    :param fun_type: 变化率计算方式（ 1：线性拟合斜率；2：最大值和最小值差值）
    :return: 总体变化率(float) 和 时间窗内的变化率(numpy.ndarray)，其单位为每分钟的变化率
    '''
    values = kpi_data['监控项的值']
    times = pd.to_datetime(kpi_data['日期'])
    value_df = pd.DataFrame(values.values.tolist(), index=times)

    if fun_type == 1:
        expanding_seconds = (times.iloc[-1] - times.iloc[0]).days
        expanding_minutes = expanding_seconds / 1
        # 拟合斜率-总体变化率
        k, b = optimize.curve_fit(f_1, np.arange(
                                      kpi_data.shape[0]), values)[0]
        value_df = value_df.rolling(window, min_periods=window).apply(
            lambda x: curve_fit_fun(x) / (expanding_minutes / len(values)))

    elif fun_type == 2:
        expanding_seconds = (times[np.argmax(values)] - times[np.argmin(values)]).days
        expanding_minutes = expanding_seconds / 1
        # 最大值和最小值差值-总体变化率
        k = (max(values) - min(values)) / (np.argmax(values) - np.argmin(values))
        value_df = value_df.rolling(window, min_periods=window).apply(
            lambda x: min_max_fit_fun(x))
    value_df['日期'] = value_df.index
    value_df['监控项的值'] = value_df[0]
    return k, value_df[['日期', '监控项的值']]


# 统计信息功能，根据输入数据计算不同时间单位的均值和极值
def Statistics(kpi_data):
    '''
    :param kpi_data: 进行变化率计算的数据，dataframe格式，包括“日期”和“监控项的值”
    :return: 数据统计分析结果：statistics_dict
        类StatisticsDict，包含hour_data, day_data, week_data, month_data;
                class StatisticsDict:
                {
                    hour_data: None
                    day_data: None
                    week_data: None
                    month_data: None
                }
        每个元素为dataframe格式，包含index(日期)、mean(单位时间均值)、min(单位时间最小值)、max(单位时间最大值);
                                          mean        min        max
                日期
                2021-06-21 03:00:00   9.019167   1.462500  13.220000
                2021-06-21 04:00:00 -12.223148 -25.900000  11.675000
                2021-06-21 05:00:00  -2.492593 -27.433333  14.855556
                2021-06-21 06:00:00  14.570486  12.037500  15.566667
                2021-06-21 07:00:00   5.638095  -0.637500  14.837500
    '''
    kpi_data = kpi_data[['日期', '监控项的值']]
    kpi_data['日期'] = pd.to_datetime(kpi_data['日期'])
    statistics_dict = StatisticsDict()
    hour_statistics = groupby_fun(kpi_data, 'H')
    day_statistics = groupby_fun(kpi_data, 'D')
    week_statistics = groupby_fun(kpi_data, '7D')
    month_statistics = groupby_fun(kpi_data, 'M')
    statistics_dict.__dict__.update({"hour_data": hour_statistics,
                                     'day_data': day_statistics,
                                     'week_data': week_statistics,
                                     'month_data': month_statistics})
    return statistics_dict


# 移动平均功能，根据用于输入的窗口大小进行滑动计算
def moving_average(kpi_data, window):
    '''
    :param kpi_data: 进行移动平均的数据，dataframe格式，包括“监控项的值”和"日期"
    :param window: 用户输入的时间框大小（点的个数）
    :return: 移动平均后的数据（dataframe格式，包含"日期"和"监控项的值"，前window-1个为None）
    '''
    df = kpi_data['监控项的值'].rolling(window, min_periods=window).mean()
    return df


# 分组功能
def groupby_fun(data, unit):
    statistics = pd.DataFrame()
    statistics['mean'] = data.groupby(pd.Grouper(key='日期', freq=unit)).mean()
    statistics['min'] = data.groupby(pd.Grouper(key='日期', freq=unit)).min()
    statistics['max'] = data.groupby(pd.Grouper(key='日期', freq=unit)).max()
    return statistics


# 统计单位类定义
class StatisticsDict:
    hour_data: None
    day_data: None
    week_data: None
    month_data: None


# 变化率计算方式1：拟合函数斜率
def curve_fit_fun(values):
    k, b = optimize.curve_fit(f_1, np.arange(len(values)), values)[0]
    return k


# 变化率计算方式2：最大最小值的差值
def min_max_fit_fun(values):
    times = pd.to_datetime(values.index)
    expanding_seconds = abs((times[np.argmax(values)] - times[np.argmin(values)]).days)
    expanding_minutes = expanding_seconds / 1
    k = (max(values) - min(values))
    return k / (expanding_minutes / (np.argmax(values) - np.argmin(values)))


# 一元线性回归函数
def f_1(x, A, B):
    return A*x + B


if __name__ == '__main__':
    excel = 2
    if excel == 1:
        file = r'高一线-精轧区域-BGV机组-26#锥箱-加速度有效值-三个月数据.xlsx'
        title = '26#锥箱-传动轴输出侧轴向-加速度有效值'
    else:
        file = r'高一线-精轧区域-BGV机组-26#锥箱-速度有效值-三个月数据.xlsx'
        title = '26#锥箱-传动轴输出侧轴向-速度有效值'
    df = pd.read_excel(file, header=0)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df[['日期', '监控项的值']]


    # plt.plot(df['日期'], df['监控项的值'])


    # 去噪点
    # res_df = rolling_filter(df, 50, 1)
    # res_df['日期'] = pd.to_datetime(res_df['日期'])


    # 移动平均
    # new_res_df = moving_average(df, 20)
    # new_res_df = moving_average(res_df, 20)

    # df['监控项的值'] = new_res_df[1]
    # d = {
    #     '日期': df['日期'].to_list(),
    #     '监控项的值': new_res_df.to_list()
    # }
    # new_df = pd.DataFrame(d)
    # new_df.drop(new_df[np.isnan(new_df['监控项的值'])].index, inplace=True)
    #
    # 日，最大值
    statistics = groupby_fun(df, 'D')
    # print(statistics.index)
    new_df = {
        '日期': statistics.index.to_list(),
        '监控项的值': statistics['max'].to_list()
    }
    new_df = pd.DataFrame(new_df)
    new_df.drop(new_df[np.isnan(new_df['监控项的值'])].index, inplace=True)


    # 变化率
    k, res_res_df = changing_rate(new_df, 20, 2)

    title = title + '(%s)'% k


    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.legend(['日-最大值', '变化率'])
    ax.set_ylabel('日-最大值')
    ax.set_xlabel('日期')
    line1 = ax.plot(new_df['日期'], new_df['监控项的值'], label='日-最大值')

    ax1 = ax.twinx()

    line2 = ax1.plot(new_df['日期'], res_res_df['监控项的值'], color='r', label='变化率')
    ax1.set_ylabel('变化率')

    lines = line1 + line2
    labels = [label.get_label for label in lines]

    ax.legend(lines, ['日-最大值', '变化率'])

    plt.title(title)
    plt.grid()
    plt.show()


