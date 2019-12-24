import numpy as np
import json

class Pfpr:
    """
    False Positive Rate
    UCID
    """
    def __init__(self):
        pass

    def run(self, blocksize=64, t_list=[0.9888]):
        with open('modules/figure4/ucid/ucid_cc_dict_%d.json' % blocksize, 'r') as f:
            ucid_cc_dict = json.loads(f.read())

        ucid_cc_list = np.array(ucid_cc_dict['cc_list'])

        result_list = []
        for T in t_list:
            denominator = ucid_cc_list.size
            numerator = ucid_cc_list[ucid_cc_list > T].size
            result = numerator / denominator 
            result = result
            result_list.append(result)
        return list(result_list)

class Ptpr(object):
    """
    True Positive Rate
    Copydays
    """
    def __init__(self):
        pass

    def run(self, blocksize=64, t_list=[0.9888]):
        with open('modules/figure4/copydays/copydays_cc_dict_%d.json' % blocksize, 'r') as f:
            copydays_cc_dict = json.loads(f.read())

        numerator = 0
        denominator = 0
        copydays_cc_list = []
        for d in copydays_cc_dict.values():
            cc_list = d['cc_list']
            copydays_cc_list = np.append(copydays_cc_list, cc_list)

        result_list = []
        for T in t_list:
            denominator = copydays_cc_list.size
            numerator = copydays_cc_list[copydays_cc_list > T].size
            result = numerator / denominator 
            result = result
            result_list.append(result)

        return list(result_list)
