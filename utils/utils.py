import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


def dict_html(dict_obj, current_time):
    """
    将给定的字典对象格式化为 HTML 表格，用于显示模型参数。

    参数:
    - dict_obj: 包含模型参数的字典对象，key 为参数名称，value 为参数值。
    - current_time: 当前时间，用于在 HTML 中显示。

    返回:
    - output: 格式化后的 HTML 字符串。
    """
    out = ''  # 初始化空字符串，用于存储 HTML 内容

    # 遍历字典中的所有键值对
    for key, value in dict_obj.items():

        # 过滤掉不需要展示的部分
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold']:
            continue  # 如果当前的 key 在不需要显示的列表中，则跳过当前的项

        # 对于需要展示的参数，构造 HTML 表格的每一行
        out += f'<tr><td>{key}</td><td>{value}</td></tr>'

    # 最终输出的 HTML 内容，包含表格头部和表格内容
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output


