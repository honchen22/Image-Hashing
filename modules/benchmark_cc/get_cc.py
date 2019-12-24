import numpy as np
import json

class CorrelationCoefficient:
    def __init__(self):
        self.hash_dict = {}
        self.cc_dict = {}

    def init_hash_dict(self):
        with open('modules/benchmark_cc/standard_benchmark_hash.json', 'r') as f:
            self.hash_dict = json.loads(f.read())

    def init_cc_dict(self):
        with open('modules/benchmark_cc/cc_format.json', 'r') as f:
            self.cc_dict = json.loads(f.read())

    def run(self):
        self.init_hash_dict()
        self.init_cc_dict()

        for image_name in self.cc_dict:
            image = self.cc_dict[image_name]
            for attack_type in image:
                self.set_cc_dict(image_name, attack_type)

        with open('modules/benchmark_cc/standard_benchmark_cc_dict.json', 'w') as f:
            content = json.dumps(self.cc_dict, indent=2)
            content = content.replace("'", '"').replace('\n        ', '')
            f.write(content)

    def set_cc_dict(self, image_name='Airplane', attack_type='JPEG'):
        image = self.hash_dict[image_name]
        attack = image[attack_type]
        param_range = attack['param_range']

        param_cnt = attack['param_cnt']
        print(image_name, attack_type, param_cnt, param_range)

        cc_list = [0] * param_cnt

        H0 = image['raw']['H']
        for i in range(param_cnt):
            H_key = 'H%d' % (i + 1)
            Hi = attack[H_key]
            cc = np.corrcoef(H0, Hi)[0][1]
            cc_list[i] = cc

        self.cc_dict[image_name][attack_type]['cc_list'] = cc_list

# if __name__ == '__main__':
#     cc = CorrelationCoefficient()
#     cc.run()