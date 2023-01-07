# -*- coding: utf-8 -*-
# @Author  : zhangtao
# @File    : Calculate_time.py
# @Desc    : 两数据项时间模糊对齐，可设置允许误差，单位分钟
# @Time    : 2022/11/24 15:32
# @Software: PyCharm

import numpy as np
import pandas as pd
from Input_Datatype import Dataset

class Calculate_time:
    def __init__(self):
        self.model_dict = {
            "=": '==',
            ">": ">",
            "≥": ">=",
            "<": "<",
            "≤": "<=",
        }

    def execute(self, A1: Dataset = None, A2: Dataset = None, C1: int or float = 1, mode: str = '<', result: dict = None):
        """
        两参数时间对齐，可设置允许误差，在误差内即使两参数的时间点不同也可视为同一时间点，默认完全匹配。
        :param mode:    str 条件
        :param C1:      int or float 误差值
        :param A1:      Dataset 参数A
        :param A2:      Dataset 参数B
        :param result:  dict 返回结果
        :return:
        """
        out1 = Dataset.create_dataset_from_dataset(A1)
        out2 = Dataset.create_dataset_from_dataset(A2)

        data_1 = A1.data
        data_2 = A2.data

        if isinstance(A1, Dataset) and isinstance(A2, Dataset):
            time_in1 = A1.time.copy()
            time_in2 = A2.time.copy()

            data_A = dict()
            data_A['date'] = A1.time
            data_A['data_A'] = data_1

            data_B = dict()
            data_B['date'] = A2.time
            data_B['data_B'] = data_2

            concat_result = self.concat_time(data_A, data_B)
            filter_data = self.filter_diff(C1=C1, mode=mode, data_df=concat_result)
            out1.time = list(filter_data['date'])
            out1.data = list(filter_data['data_A'])
            out2.time = list(filter_data['date'])
            out2.data = list(filter_data['data_B'])

            # 判断是否为空, 若为空则取各自数据的第一个值
            if not out1.time:
                # 若没有对应相同的时间点, 则输出两个数据中第一个点中较小的时间点
                if time_in1[0] < time_in2[0]:
                    out1.time = [str(time_in1[0][:23])]
                    out2.time = [str(time_in1[0][:23])]
                else:
                    out1.time = [str(time_in2[0][:23])]
                    out2.time = [str(time_in2[0][:23])]

                out1.start_time = [str(A1.start_time)[:23]]
                out1.data = np.array([data_1.data[0]])

                out2.start_time = [str(A2.start_time)[:23]]
                out2.data = np.array([data_2.data[0]])


        result['Out1'] = out1
        result['Out2'] = out2

    def concat_time(self, data_A, data_B):
        """
        两参数时间对齐
        :param data_A: dict 参数A
        :param data_B: dict 参数B
        :return:
        """
        df_A = pd.DataFrame(data_A)
        df_B = pd.DataFrame(data_B)

        if df_A.shape[0] < df_B.shape[0]:
            df_A['date_time'] = pd.to_datetime(df_A['date'])
            df_B['date_time'] = pd.to_datetime(df_B['date'])
            df_B['passed_time'] = df_B['date_time']

            df = pd.merge_asof(df_A, df_B[['date_time', 'passed_time', 'data_B']], on='date_time', direction="nearest")
        else:
            df_B['date_time'] = pd.to_datetime(df_B['date'])
            df_A['date_time'] = pd.to_datetime(df_A['date'])
            df_A['passed_time'] = df_A['date_time']

            df = pd.merge_asof(df_B, df_A[['date_time', 'passed_time', 'data_A']], on='date_time', direction="nearest")

        df['date_diff'] = abs(round((df['date_time'] - df['passed_time']) / pd.Timedelta("1 minutes"), 2))
        return df

    def filter_diff(self, C1, mode, data_df):
        """
        时间误差，根据参数 C1 过滤
        :param data_df: DataFarm     时间对齐后数据
        :param mode:    str          条件
        :param C1:      int or float 误差值
        :return:
        """
        query = 'date_diff %s %s'% (self.model_dict[mode], C1)
        filter_df = data_df.query(query)
        return filter_df