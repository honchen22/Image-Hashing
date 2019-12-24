import numpy as np
import json

class CorrelationCoefficientFigure3:
    def __init__(self):
        self.ucid_hash_dict = {}
        self.cc_dict = {'cc_list': []}

    def run(self):
        # 
        with open('modules/figure3/ucid_hash_dict_64.json', 'r') as f:
            self.ucid_hash_dict = json.loads(f.read())

        hash_list = list(self.ucid_hash_dict.values())
        hash_list_length = len(hash_list)
        # print(hash_list_length)
        cc_list = [0] * (hash_list_length * (hash_list_length - 1) // 2)
        cnt = 0
        for i in range(hash_list_length - 1):
            for j in range(i + 1, hash_list_length):
                h1, h2 = hash_list[i], hash_list[j]
                cc_list[cnt] = np.corrcoef(h1, h2)[0][1]
                cnt += 1
        self.cc_dict['cc_list'] = cc_list

        with open('modules/figure3/ucid_cc_dict_64.json', 'w') as f:
            content = json.dumps(self.cc_dict, indent=2)
            content = content.replace("'", '"').replace('\n    ', ' ')
            f.write(content)

'''
if __name__ == '__main__':
    cc = CorrelationCoefficientFigure3()
    cc.run()
'''