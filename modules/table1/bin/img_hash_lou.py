#! -*- coding:utf-8 -*-

"""
sudo apt-get install opencv-python
sudo pip3 install opencv-python

ucid: http://jasoncantarella.com/downloads/
wget http://jasoncantarella.com/downloads/ucid.v2.tar.gz

copydays: http://lear.inrialpes.fr/people/jegou/data.php#copydays
wget http://pascal.inrialpes.fr/data/holidays/copydays_original.tar.gz
"""

import cv2 as cv
import numpy as np
from math import cos, sqrt
from math import pi as PI

IMAGESIZE = 512
BLOCKSIZE = 64
GRAYLEVEL = 256
M_PI = 3.1415926

class MyImage:
    def __init__(self, filename, q):
        img = cv.imread(filename)
        img = cv.resize(img, (IMAGESIZE, IMAGESIZE))
        self.im = np.zeros(shape=(IMAGESIZE, IMAGESIZE), dtype=int)

        for i in range(IMAGESIZE):
            for j in range(IMAGESIZE):
                r, g, b = map(int, img[i][j])
                self.im[i][j] = (r + g + b) / (3 * q)

    def get_intensity_matrix(self):
        return self.im

class MyGLCM:
    def __init__(self, im, theta):
        self.ttl_val = 0
        self.row_mean = 0
        self.col_mean = 0
        self.row_stdev = 0
        self.col_stdev = 0
        self.matrix = np.zeros(shape=(GRAYLEVEL, GRAYLEVEL))
        self.set_glcm_params(im, theta)

    def set_glcm_params(self, im, theta):
        t_matrix = np.zeros(shape=(GRAYLEVEL, GRAYLEVEL))
        if theta == 0:
            for row in range(IMAGESIZE):
                for col in range(IMAGESIZE - 1):
                    x = im[row][col]
                    y = im[row][col + 1]
                    self.matrix[x][y] += 1
                    t_matrix[y][x] += 1
        elif theta == 45:
            for row in range(1, IMAGESIZE):
                for col in range(IMAGESIZE - 1):
                    x = im[row][col]
                    y = im[row - 1][col + 1]
                    self.matrix[x][y] += 1
                    t_matrix[y][x] += 1
        elif theta == 90:
            for row in range(1, IMAGESIZE):
                for col in range(IMAGESIZE):
                    x = im[row][col]
                    y = im[row - 1][col]
                    self.matrix[x][y] += 1
                    t_matrix[y][x] += 1
        elif theta == 135:
            for row in range(1, IMAGESIZE):
                for col in range(1, IMAGESIZE - 1):
                    x = im[row][col]
                    y = im[row - 1][col - 1]
                    self.matrix[x][y] += 1
                    t_matrix[y][x] += 1

        # set glcm params
        
        
        # '''
        self.ttl_val = 0
        tmp_sum = np.zeros(shape=GRAYLEVEL)
        self.row_mean = 0
        for x in range(GRAYLEVEL):
            s = np.sum(self.matrix[x])
            tmp_sum[x] = s
            self.ttl_val += s
            self.row_mean += (x + 1) * s
        self.row_mean /= self.ttl_val

        row_var = 0
        for x in range(GRAYLEVEL):
            row_var += pow(x + 1 - self.row_mean, 2) * tmp_sum[x]
        row_var /= self.ttl_val
        self.row_stdev = sqrt(row_var)

        self.col_mean = 0
        for y in range(GRAYLEVEL):
            s = np.sum(t_matrix[y])
            tmp_sum[y] = s
            self.col_mean += (y + 1) * s
        self.col_mean /= self.ttl_val

        col_var = 0
        for y in range(GRAYLEVEL):
            col_var += pow(y + 1 - self.col_mean, 2) * tmp_sum[y]
        col_var /= self.ttl_val
        self.col_stdev = sqrt(col_var)
        # '''

    def get_texture_features(self, im, T, theta):
        contrast = 0
        correlation = 0
        energy = 0
        homogeneity = 0
        for x in range(GRAYLEVEL):
            for y in range(GRAYLEVEL):
                p = self.matrix[x][y]
                numerator = (x - self.row_mean) * (y - self.col_mean)
                denominator = self.row_stdev * self.col_stdev
                if denominator != 0:
                    correlation += p * numerator / denominator

                contrast += p * pow(x - y, 2)
                energy += pow(p, 2)
                homogeneity += p / (1 + pow(x - y, 2))

        contrast /= self.ttl_val
        correlation /= self.ttl_val
        energy /= pow(self.ttl_val, 2)
        homogeneity /= self.ttl_val

        offset = (theta // 45 ) * 4
        T[offset] = contrast
        T[offset + 1] = correlation
        T[offset + 2] = energy
        T[offset + 3] = homogeneity

class GlobalFeature:
    def __init__(self, im):
        self.T = np.zeros(shape=16)

        glcm0 = MyGLCM(im, 0)
        glcm0.get_texture_features(im, self.T, 0)

        glcm1 = MyGLCM(im, 45)
        glcm1.get_texture_features(im, self.T, 45)

        glcm2 = MyGLCM(im, 90)
        glcm2.get_texture_features(im, self.T, 90)

        glcm3 = MyGLCM(im, 135)
        glcm3.get_texture_features(im, self.T, 135)

        # self.show_T()

    def show_T(self):
        for i in range(4):
            for j in range(4):
                print(self.T[i*4 + j], end=', ')
            print()

class LocalFeature:
    def __init__(self, im):
        self.num_of_blocks_in_a_row = IMAGESIZE // BLOCKSIZE
        self._N = int(pow(self.num_of_blocks_in_a_row, 2))
        self._2n = BLOCKSIZE
        self._n = BLOCKSIZE // 2
        self.sqrt_2_div_by_s = sqrt(2) / BLOCKSIZE
        self.PI_div_by_2s = M_PI / (2 * BLOCKSIZE)
        self.Q = np.zeros(shape=(self._2n, self._N))

        for i in range(self._N):
            row_begin = int((i // self.num_of_blocks_in_a_row) * BLOCKSIZE)
            col_begin = int((i % self.num_of_blocks_in_a_row) * BLOCKSIZE)

            # print('Qi begin %d' % i)
            Bi = im[row_begin:row_begin+BLOCKSIZE, col_begin:col_begin+BLOCKSIZE]
            imf = np.float32(Bi)/255.0
            dst = cv.dct(imf)

            for v in range(self._n):
                self.Q[v][i] = dst[0][v + 1]

            for u in range(self._n):
                self.Q[u + self._n][i] = dst[u + 1][0]

        self.normalization()

    def normalization(self):
        for r in range(self._2n):
            _mean = np.mean(self.Q[r])
            _std = np.std(self.Q[r])
            for c in range(self._N):
                self.Q[r][c] = (self.Q[r][c] - _mean) / _std

        self.set_D()

    def set_D(self):
        U0 = np.zeros(self._2n)
        for r in range(self._2n):
            U0[r] = np.mean(self.Q[r])

        self.D = np.zeros(self._N)
        for i in range(self._N):
            di = np.subtract(self.Q[:, i], U0)
            di = np.square(di).sum()
            di = np.sqrt(di)
            self.D[i] = di
        # print(self.D)

class MyHash:
    def get_img_hash(self, filename, q=1):
        img = MyImage(filename, q)
        global_feature = GlobalFeature(img.im)
        local_feature = LocalFeature(img.im)
        H = np.zeros(80, dtype=int)
        for i in range(16):
            val = global_feature.T[i]
            H[i] = round(val * 10 + 0.5)
        for i in range(64):
            val = local_feature.D[i]
            H[16 + i] = round(val * 10 + 0.5)
        return H

    def get_img_cc(self, filename1='airplane.tiff', filename2='house512.tiff'):
        # print(filename1, filename2)
        H1 = self.get_img_hash(filename1, 2)
        H2 = self.get_img_hash(filename2, 2)

        print(np.corrcoef(H1, H2))

def get_h(filename, q=1):
    img = MyImage(filename, q)
    global_feature = GlobalFeature(img.im)
    local_feature = LocalFeature(img.im)
    H = np.zeros(80)
    for i in range(16):
        val = global_feature.T[i]
        H[i] = round(val * 10 + 0.5)
    for i in range(64):
        val = local_feature.D[i]
        H[16 + i] = round(val * 10 + 0.5)
    return list(H)


def run():
    file_list = os.listdir('ucid')
    hash_dict = {}

    for filename in file_list[:1]:
        try:
            h = get_h('ucid/' + filename)
            filename = filename[:-4]
            hash_dict[filename] = h
        except Exception as e:
            pass

    with open('ucid_hash_dict.txt', 'w') as f:
        s = json.dumps(hash_dict).replace("'", '"')
        f.write(s)

if __name__ == '__main__':
    run()
