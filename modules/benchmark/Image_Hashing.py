import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


class Preprocessing:
    def __init__(self, S, s):
        self.pi = 3.1415926
        self.Colorlevel = 256
        self.S = S
        self.s = s

    def run(self, filepath):
        S = self.S
        return self.get_input_img(filepath, S)

    def rgb2hsi(self, rgbImg):
        #np.seterr(divide='ignore', invalid='ignore')
        rgbImg = rgbImg.astype(np.float32)
        # 归一化到[0,1]
        img_bgr = rgbImg.copy()/255
        b, g, r = cv2.split(img_bgr)

        # Tdata = np.where((2*np.sqrt((r-g)**2+(r-b)*(g-b))) != 0,np.arccos((2*r-b-g)/(2*np.sqrt((r-g)**2+(r-b)*(g-b)))),0)
        # Hdata = np.where(g >= b,Tdata,2*self.pi-Tdata)
        # Hdata = Hdata / (2*self.pi)
        # Sdata = np.where((b+g+r) != 0, 1 - 3*np.minimum(b,g,r)/(b+g+r),0)
        Idata = (b+g+r)/3

        # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2self.pi]之间，S和I在[0,1]之间
        hsiImg = rgbImg.copy()
        # hsiImg[:,:,0] = Hdata*255
        # hsiImg[:,:,1] = Sdata*255
        hsiImg[:, :, 2] = Idata*255

        return hsiImg

    def get_input_img(self, inpath, S):
        '''
        :param rgbImg:输入图片路径
        :return i
        '''
        # 提取I分量作为输入图像
        rgbImg = cv2.imread(inpath)
        rgbImg = cv2.resize(rgbImg, (S, S))
        hsiImg = self.rgb2hsi(rgbImg)
        input_img = np.zeros(
            [hsiImg.shape[0], hsiImg.shape[1], 1], dtype=np.uint8)
        for i in range(hsiImg.shape[0]):
            for j in range(hsiImg.shape[1]):
                input_img[i, j, 0] = hsiImg[i, j, 2]
        return input_img


class Global_Statistical_Feature:
    # 图像矩阵原点在左上角，所以di,dj 会有点绕
    # 此处p矩阵的计算做了变种，没有完全按公式来，要不时间复杂度将是O(256*256*O(512*512)) = 150亿数量级的！
    def __init__(self, S=512, s=64):
        self.d = 1
        self.S = S
        self.s = s
        self.N = (self.S//self.s)**2
        self.n = self.s//2
        self.theta = 0
        self.Colorlevel = 256

    def run(self, input_img):
        Colorlevel = self.Colorlevel
        d = self.d
        theta = self.theta
        S = self.S
        p = self.get_P_Matrix(Colorlevel, input_img, d, theta, S)
        contrast = self.get_Contrast(p)
        correlation = self.get_Correlation(p)
        energy = self.get_Energy(p)
        homegeneity = self.get_Homegeneity(p)
        self.T = self.get_T(Colorlevel, input_img, d, S)
        #assert len(self.T) == 16
        return self.T

    def get_P_Matrix(self, Colorlevel, input_img, d, theta, S):
        """GLCM计算.

        :param x,y: 为0-255的像素值
        :param d: 距离
        :param theta: 方向

        :return p:归一化的P矩阵
        :rtype ndarray(256*256)
        """
        di, dj = 0, 0

        if theta == 0:
            dj = d
        elif theta == 45:
            di, dj = -d, d
        elif theta == 90:
            di = -d
        elif theta == 135:
            di, dj = -d, -d

        rows, cols, channels = input_img.shape
        # assert rows == S
        # assert cols == S
        # assert channels == 1
        p = np.zeros((Colorlevel, Colorlevel))

        for i in range(rows):
            if i + di < 0 or i + di >= rows:
                continue
            for j in range(cols):
                if j + dj < 0 or j + dj >= cols:
                    continue

                self.pixelx = input_img[i][j][0]
                self.pixely = input_img[i+di][j+dj][0]
                p[self.pixelx, self.pixely] += 1

        p = p/p.sum()
        return p

    def get_Contrast(self, p):
        '''
        :param p: P矩阵

        :return contrast:对比度
        :rtype float
        '''
        contrast = 0

        for x in range(len(p)):
            for y in range(len(p[0])):
                contrast += (x-y)**2 * p[x, y]

        return contrast

    def get_Correlation(self, p):
        '''
        :param p:P矩阵

        :return correlation: 相关度
        :rtype float
        '''
        correlation = 0

        px = np.sum(p, axis=1)
        py = np.sum(p, axis=0)

        x = np.array(range(len(p)))
        ux = np.dot(x, px)

        y = np.array(range(len(p[0])))
        uy = np.dot(y, py)

        sigmax = np.dot((x-ux)**2, px)
        sigmax = np.sqrt(sigmax)

        sigmay = np.dot((y-uy)**2, py)
        sigmay = np.sqrt(sigmay)

        for x in range(len(p)):
            for y in range(len(p[0])):
                if (sigmax * sigmay) == 0:
                    continue
                correlation += ((x-ux)*(y-uy)*p[x, y])/(sigmax * sigmay)
                # print(correlation)

        return correlation

    def get_Energy(self, p):
        '''
        :param p:P矩阵

        :return energy:能量
        :rtype float
        '''
        energy = 0

        energy = np.sum(p.copy()**2)
        return energy

    def get_Homegeneity(self, p):
        '''
        :param p:P矩阵

        :return homegeneity:同质性
        :rtype float
        '''
        homegeneity = 0
        for x in range(len(p)):
            for y in range(len(p[0])):
                homegeneity += p[x, y]/(1+(x-y)**2)

        return homegeneity

    def get_T(self, Colorlevel, input_img, d, S):
        '''
        :param Colorlevel:颜色量级
        :param d: 距离d

        :return T: theta分别从0到135，4个上述特征的集合
        :rtype list
        '''
        T = []
        for theta in [0, 45, 90, 135]:
            p = self.get_P_Matrix(Colorlevel, input_img, d, theta, S)
            contrast = self.get_Contrast(p)
            T.append(contrast)
            correlation = self.get_Correlation(p)
            T.append(correlation)
            energy = self.get_Energy(p)
            T.append(energy)
            homegeneity = self.get_Homegeneity(p)
            T.append(homegeneity)

        return T


class Local_Invariant_Feature:
    def __init__(self, S=512, s=64):
        self.d = 1
        self.S = S
        self.s = s
        self.N = (self.S//self.s)**2
        self.n = self.s//2
        self.theta = 0
        self.Colorlevel = 256

    def run(self, input_img):
        N = self.N
        s = self.s
        S = self.S
        B = self.get_B(input_img,S,s,N)
        C = self.get_C_Matrix(N, B)
        Q = self.get_Q(C, s, N)
        U = self.get_U(Q, N)
        D = self.caculate_Euclidean_Distance(U)
        # len(D) == 64
        return D

    def get_B(self, input_img,S,s,N):
        '''# 将图像分成s*s的块 , 
        :param input_img: 输入图像
        :param s: 输入图像分割成好多个块，每个块是s*s的像素

        :return B:分割后并索引好的块矩阵
        :rtype list
        '''
        S = input_img.shape[0]
        row = S//s
        col = row
        B = [0]*N
        for i in range(N):
            pos_x = (i//row)*s
            pos_y = (i % row)*s
            B[i] = input_img[pos_x:pos_x+s, pos_y:pos_y+s, 0]
            #assert B[i].shape == (s, s)

        return B

    def DCT_transformation(self, img_src):
        """# DCT 变换.

        :param img_src:源图像

        :return img_dct:dct变换后的图像
        :rtype ndarray(64*64)
        """
        img_src = np.float32(img_src)  # 将数值精度调整为32位浮点型
        img_dct = cv2.dct(img_src)  # 使用dct获得img的频域图像
        return img_dct

    def get_C_Matrix(self, N, B):
        '''
        :param N:分割的块数
        :param B:分割后的块矩阵

        :return C:DCT变换后的C矩阵
        :rtype list
        '''
        C = [0]*N
        for i in range(N):
            C[i] = self.DCT_transformation(B[i])
        return C

    def get_Q(self, C, s, N):
        """# 获取Q向量.

        :param C: DCT变换后的C矩阵
        :param s: s*s 的小块
        :param N: 分割的块数

        :return Q: C[0,1]～C[0,n],C[1,0]~C[n,0] 组合后的向量
        :rtype ndarray(64*64)
        """
        n = s//2
        Q = [0]*N
        for i in range(N):
            Q[i] = []
            for v in range(n):
                Q[i].append(C[i][0, v])
            for u in range(n):
                Q[i].append(C[i][u, 0])

            Q[i] = np.array(Q[i], dtype='float32')
            
        Q = np.array(Q, dtype='float32').T
        return Q

    def get_U(self, Q, N):
        """# 获取U向量.

        :param Q: C[0,1]～C[0,n],C[1,0]~C[n,0] 组合后的向量
        :param N: 分割后的块数

        :return U: 归一化Q
        :rtype ndarray(64*64)
        """
        # 归一化Q
        U = np.zeros(Q.shape)
        for i in range(len(Q)):
            amax = Q[i, :].max()
            amin = Q[i, :].min()
            U[i] = (Q[i] - amin) / (amax-amin)

        return U

    def caculate_Euclidean_Distance(self, U):
        """# 计算U向量内部的欧式距离.

        :param U: 归一化的Q

        :return d: U向量内部的欧式距离
        :rtype list
        """
        # U0是每一行的均值，U的其他不变, U0.shape = (64,1)
        U0 = np.mean(U, axis=1)
        #print(U.shape,U0.shape)
        #U0 = np.resize(U0,(-1,1))
        #print(U.shape,U0.shape)
        d = [0]*U.shape[1]
        for i in range(len(d)):
            #print(U[:,i])
            d[i] = np.sqrt(np.sum((U[:, i] - U0)**2))

        return d


class Quantization:
    def __init__(self, T, D):
        self.T = T
        self.D = D

    def run(self):
        T = self.T
        D = self.D
        H1 = self.get_H1(T, D)
        #assert len(H1) == 80

        h = self.quantize(H1)
        return h

    def get_H1(self, T, D):
        """# 将T和D 特征向量 组合成H1.

        :param T: 16个Global Statistical Feature 特征
        :param D: 64个Local Invariant Feature 特征

        :return H1: [T D]
        :rtype list
        """
        H1 = [] + T + D
        return H1

    def quantize(self, H1):
        '''# quantize H1 to integer h(q) , hash h = [h(q)] (1<=q<=80)
        :param H1: [T D]

        :return h: 量化后的H1
        :rtype list
        '''
        H1 = np.array(H1)
        h = np.around(H1*10 + 0.5)
        return list(h)


class Image_Hash:
    def __init__(self, S=512, s=64):
        self.s = s
        self.S = S

    def run(self, filepath):
        s = self.s
        S = self.S
        input_img = Preprocessing(S=S, s=s).run(filepath)
        T = Global_Statistical_Feature(S=S, s=s).run(input_img)
        D = Local_Invariant_Feature(S=S, s=s).run(input_img)
        h = Quantization(T, D).run()
        return h

    def get_h(self, filepath):
        return self.run()
