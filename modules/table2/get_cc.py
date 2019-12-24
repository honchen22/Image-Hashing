import numpy as np
import json

class Pfpr:
    """
    False Positive Rate
    UCID
    """
    def __init__(self):
        pass

    def run(self, t_list=[0.9888]):
        # with open('modules/table2/ucid_cc_dict_Hong_div10.json', 'r') as f:
        with open('modules/table2/ucid_cc_dict_64.json', 'r') as f:
            ucid_cc_dict = json.loads(f.read())

        ucid_cc_list = np.array(ucid_cc_dict['cc_list'])

        result_list = []
        for T in t_list:
            denominator = ucid_cc_list.size
            numerator = ucid_cc_list[ucid_cc_list > T].size
            result = 100 * numerator / denominator 
            result = '%.2f%%' % result
            result_list.append(result)
        return result_list

class Ptpr(object):
    """
    True Positive Rate
    Copydays
    """
    def __init__(self):
        pass

    def run(self, t_list=[0.9888]):
        # with open('modules/table2/copydays_cc_dict_Hong.json', 'r') as f:
        with open('modules/table2/copydays_cc_dict_64.json', 'r') as f:
            copydays_cc_dict = json.loads(f.read())

        numerator = 0
        denominator = 0
        copydays_cc_list = []
        for d in copydays_cc_dict.values():
            cc_list = d['cc_list']
            copydays_cc_list = np.append(copydays_cc_list, cc_list)
        # copydays_cc_list = np.array(copydays_cc_list)

        result_list = []
        for T in t_list:
            denominator = copydays_cc_list.size
            numerator = copydays_cc_list[copydays_cc_list > T].size
            result = 100 * numerator / denominator 
            result = '%.2f%%' % result
            result_list.append(result)

        return result_list


# b = [[1, 2], [3, 4]]
# arr = np.array(b)
# print(len(arr[arr>1]))
