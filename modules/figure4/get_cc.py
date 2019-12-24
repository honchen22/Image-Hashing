import numpy as np
import json

class CopydaysHash2CCParser:
    def __init__(self):
        self.copydays_hash_dict = {}
        self.cc_dict = {
            'Brightness': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Contrast': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Gamma': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Image_Scaling': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'JPEG': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Rotation': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Salt_and_pepper_noise': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            },
            'Speckle_noise': {
                'maximum': 1,
                'minimum': -1,
                'mean': 0,
                'cc_list': []
            }
        }
        self.hash_param_format_dict = {
            'JPEG': {
                "param_range": [30, 40, 50, 60, 70, 80, 90, 100],
                "param_cnt": 8,
            },
            "Image_Scaling": {
                "param_range": [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
                "param_cnt": 6,
            },
            "Salt_and_pepper_noise":{
                "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
                "param_cnt": 10,
            },
            "Brightness": {
                "param_range": [-20, -10, 10, 20],
                "param_cnt": 4,
            },
            "Contrast": {
                "param_range": [-20, -10, 10, 20],
                "param_cnt": 4,
            },
            "Gamma": {
                "param_range": [0.75, 0.9, 1.1, 1.25],
                "param_cnt": 4,
            },
            "Gaussian": {
                "param_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "param_cnt": 8,
            },
            "Rotation": {
                "param_range": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
                "param_cnt": 10,
            },
            "Speckle_noise":{
                "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
                "param_cnt": 10,
            }
        }
        self.copydays_filename_list = [
            '212000.jpg', '203100.jpg', '203900.jpg', '200400.jpg', '207100.jpg', '209200.jpg', '210900.jpg', '203700.jpg', '212700.jpg', '206100.jpg', '203400.jpg', '213900.jpg', '201900.jpg', '211500.jpg', '200300.jpg', '203000.jpg', '204000.jpg', '214300.jpg', '213300.jpg', '206000.jpg', '209300.jpg', '200100.jpg', '214200.jpg', '211000.jpg', '200200.jpg', '207600.jpg', '201600.jpg', '204900.jpg', '200600.jpg', '202000.jpg', '212600.jpg', '208500.jpg', '210000.jpg', '212300.jpg', '215500.jpg', '204200.jpg', '207400.jpg', '213000.jpg', '208200.jpg', '213400.jpg', '210700.jpg', '207000.jpg', '213600.jpg', '204300.jpg', '208400.jpg', '207700.jpg', '207900.jpg', '206300.jpg', '209100.jpg', '208900.jpg', '205100.jpg', '214000.jpg'
            ]

    def run(self, blocksize=64):
        with open('modules/figure4/copydays/copydays_hash_dict_%d.json' % blocksize, 'r') as f:
            self.copydays_hash_dict = json.loads(f.read())

        for attack_type in self.cc_dict:
            for raw_filename in self.copydays_filename_list:
                key0 = raw_filename
                h0 = self.copydays_hash_dict[key0]

                filename = raw_filename[:-4]

                param_range = self.hash_param_format_dict[attack_type]['param_range']
                for param in param_range:
                    keyi = '%s_%s_%s_.jpg' % (filename, attack_type.lower(), str(param))
                    hi = self.copydays_hash_dict[keyi]
                    cc = np.corrcoef(h0, hi)[0][1]
                    # if cc < 0:
                    #     print(key0, keyi, cc)
                    self.cc_dict[attack_type]['cc_list'].append(cc)

            cc_list = self.cc_dict[attack_type]['cc_list']
            self.cc_dict[attack_type]['maximum'] = np.max(cc_list)
            self.cc_dict[attack_type]['minimum'] = np.min(cc_list)
            self.cc_dict[attack_type]['mean'] = np.mean(cc_list)

        with open('modules/figure4/copydays/copydays_cc_dict_%d.json' % blocksize, 'w') as f:
            content = json.dumps(self.cc_dict, indent=2)
            content = content.replace("'", '"').replace('\n    ', ' ')
            f.write(content)

class UCIDHash2CCParser:
    def __init__(self):
        self.ucid_hash_dict = {}
        self.cc_dict = {'cc_list': []}

    def run(self, blocksize=64):
        with open('modules/figure4/ucid/ucid_hash_dict_%d.json' % blocksize, 'r') as f:
            self.ucid_hash_dict = json.loads(f.read())

        hash_list = list(self.ucid_hash_dict.values())
        hash_list_length = len(hash_list)
        print('ucid dataset size:', hash_list_length)
        cc_list = [0] * (hash_list_length * (hash_list_length - 1) // 2)
        cnt = 0
        for i in range(hash_list_length - 1):
            for j in range(i + 1, hash_list_length):
                h1, h2 = hash_list[i], hash_list[j]
                cc_list[cnt] = np.corrcoef(h1, h2)[0][1]
                cnt += 1
        self.cc_dict['cc_list'] = cc_list

        with open('modules/figure4/ucid/ucid_cc_dict_%d.json' % blocksize, 'w') as f:
            content = json.dumps(self.cc_dict, indent=2)
            content = content.replace("'", '"').replace('\n    ', ' ')
            f.write(content)

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
