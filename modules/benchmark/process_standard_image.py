from modules.benchmark.Image_Hashing import *
import numpy as np
import json, os

'''
hash_dict = {
    "Airplane": {
        "raw": { "H": [] },
        "JPEG": {
            "param_range": [30, 40, 50, 60, 70, 80, 90, 100],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": []
        },
        "Image_Scaling": {
            "param_range": [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
            "param_cnt": 6,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": []
        },
        "Salt_and_pepper_noise":{
            "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        },
        "Brightness": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Contrast": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": [],
        },
        "Gamma": {
            "param_range": [0.75, 0.9, 1.1, 1.25],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Gaussian": {
            "param_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
        },
        "Rotation": {
            "param_range": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        }
    },
    "Baboon": {
        "raw": { "H": [] },
        "JPEG": {
            "param_range": [30, 40, 50, 60, 70, 80, 90, 100],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": []
        },
        "Image_Scaling": {
            "param_range": [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
            "param_cnt": 6,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": []
        },
        "Salt_and_pepper_noise":{
            "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        },
        "Brightness": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Contrast": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": [],
        },
        "Gamma": {
            "param_range": [0.75, 0.9, 1.1, 1.25],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Gaussian": {
            "param_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
        },
        "Rotation": {
            "param_range": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        }
    },
    "House": {
        "raw": { "H": [] },
        "JPEG": {
            "param_range": [30, 40, 50, 60, 70, 80, 90, 100],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": []
        },
        "Image_Scaling": {
            "param_range": [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
            "param_cnt": 6,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": []
        },
        "Salt_and_pepper_noise":{
            "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        },
        "Brightness": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Contrast": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": [],
        },
        "Gamma": {
            "param_range": [0.75, 0.9, 1.1, 1.25],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Gaussian": {
            "param_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
        },
        "Rotation": {
            "param_range": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        }
    },
    "Lena": {
        "raw": { "H": [] },
        "JPEG": {
            "param_range": [30, 40, 50, 60, 70, 80, 90, 100],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": []
        },
        "Image_Scaling": {
            "param_range": [0.5, 0.75, 0.9, 1.1, 1.5, 2.0],
            "param_cnt": 6,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": []
        },
        "Salt_and_pepper_noise":{
            "param_range": [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        },
        "Brightness": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Contrast": {
            "param_range": [-20, -10, 10, 20],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": [],
        },
        "Gamma": {
            "param_range": [0.75, 0.9, 1.1, 1.25],
            "param_cnt": 4,
            "H1": [], "H2": [], "H3": [], "H4": []
        },
        "Gaussian": {
            "param_range": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "param_cnt": 8,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
        },
        "Rotation": {
            "param_range": [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5],
            "param_cnt": 10,
            "H1": [], "H2": [], "H3": [], "H4": [],
            "H5": [], "H6": [], "H7": [], "H8": [],
            "H9": [], "H10": []
        }
    }
}
'''

class BenchmarkHash2JSONParser:
    def __init__(self):
        self.hash_dict = {}
        with open('modules/benchmark/copydays_hash_format.json', 'r') as f:
            self.hash_dict = json.loads(f.read())

    def run(self):
        for image_name in self.hash_dict:
            print(image_name)
            self.set_hash_dict(image_name)
        with open('modules/benchmark/standard_benchmark_speckle_hash.json', 'w') as f:
            content = json.dumps(self.hash_dict)
            f.write(content)

    def set_hash_dict(self, image_name):
        self.set_benchmark_hash(image_name, 'tiff')

        self.set_hash(image_name, 'JPEG', 'jpg')
        self.set_hash(image_name, 'Image_Scaling', 'jpg')
        self.set_hash(image_name, 'Salt_and_pepper_noise', 'jpg')
        self.set_hash(image_name, 'Brightness', 'tif')
        self.set_hash(image_name, 'Contrast', 'tif')
        self.set_hash(image_name, 'Gamma', 'tif')
        self.set_hash(image_name, 'Gaussian', 'tif')
        self.set_hash(image_name, 'Rotation', 'tif')
        # self.set_hash(image_name, 'Speckle_noise', 'jpg')


    def set_benchmark_hash(self, image_name, image_format='tiff'):
        # try:
        basepath = 'modules/benchmark/standard/'
        filename_raw =  basepath + '%s.%s' % (image_name, image_format)
        print(filename_raw)
        image_hash = Image_Hashing()
        h0 = image_hash.get_h(filename_raw)
        self.hash_dict[image_name]['raw']['H'] = h0
        # except Exception as e:
        #     pass

    def set_hash(self, image_name='Airplane', attack_type='JPEG', image_format='tif'):
        basepath = 'modules/benchmark/standard/%s/' % attack_type
        param_range = self.hash_dict[image_name][attack_type]['param_range']
        attack_type_lower = attack_type.lower()
        i = 1
        for param in param_range:
            filename = basepath + '%s_%s_%s_.%s' % (image_name, attack_type_lower, str(param), image_format)
            print(filename)
            # try:
            image_hash = Image_Hashing()
            h = image_hash.get_h(filename)
            self.hash_dict[image_name][attack_type]['H%d' % i] = h
            # except Exception as e:
            #     print('error')
            i += 1

