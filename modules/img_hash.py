import cv2
import numpy as np
import sys

pi = 3.1415926
ColorSize = 256
d = 1
S = 512
s = 64
N = (S//s)**2
n = s//2
theta = 0

def rgb2hsi(rgb_lwpImg):
    #np.seterr(divide='ignore', invalid='ignore')
    rgb_lwpImg = rgb_lwpImg.astype(np.float32)
    # 归一化到[0,1]
    img_bgr = rgb_lwpImg.copy()/255
    b, g, r = cv2.split(img_bgr)

    # Tdata = np.where((2*np.sqrt((r-g)**2+(r-b)*(g-b))) != 0,np.arccos((2*r-b-g)/(2*np.sqrt((r-g)**2+(r-b)*(g-b)))),0)
    # Hdata = np.where(g >= b,Tdata,2*pi-Tdata)
    # Hdata = Hdata / (2*pi)
    # Sdata = np.where((b+g+r) != 0, 1 - 3*np.minimum(b,g,r)/(b+g+r),0)
    Idata = (b+g+r)/3

    # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_lwpImg = rgb_lwpImg.copy()
    # hsi_lwpImg[:,:,0] = Hdata*255
    # hsi_lwpImg[:,:,1] = Sdata*255
    hsi_lwpImg[:, :, 2] = Idata*255

    return hsi_lwpImg


def get_input_img(inpath):
    '''
    :param rgb_lwpImg:输入图片路径
    :return i
    '''
    # 提取I分量作为输入图像
    rgb_lwpImg = cv2.imread(inpath)
    rgb_lwpImg = cv2.resize(rgb_lwpImg, (512, 512))
    hsi_lwpImg = rgb2hsi(rgb_lwpImg)
    input_img = np.zeros(
        [hsi_lwpImg.shape[0], hsi_lwpImg.shape[1], 1], dtype=np.uint8)
    for i in range(hsi_lwpImg.shape[0]):
        for j in range(hsi_lwpImg.shape[1]):
            input_img[i, j, 0] = hsi_lwpImg[i, j, 2]
    return input_img

# 图像矩阵原点在左上角，所以di,dj 会有点绕
# 此处p矩阵的计算做了变种，没有完全按公式来，要不时间复杂度将是O(256*256*O(512*512)) = 150亿数量级的！


def get_P_Matrix(ColorSize, input_img, d, theta):
    '''GLCM计算

    :param x,y: 为0-255的像素值
    :param d: 距离
    :param theta: 方向

    :return p:归一化的P矩阵
    :rtype ndarray(256*256)
    '''
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
    assert rows == 512
    assert cols == 512
    assert channels == 1
    p = np.zeros((ColorSize, ColorSize))

    for i in range(rows):
        if i + di < 0 or i + di >= rows:
            continue
        for j in range(cols):
            if j + dj < 0 or j + dj >= cols:
                continue

            pixelx = input_img[i][j][0]
            pixely = input_img[i+di][j+dj][0]
            p[pixelx, pixely] += 1

    p = p/p.sum()
    return p


def get_Contrast(p):
    '''
    :param p: P矩阵

    :return contrast:对比度
    :rtype float
    '''
    contrast = 0

    for x in range(len(p)):
        for y in range(len(p[0])):
            contrast += (x-y)**2 * p[x, y]
    # contrast = contrast/10
    return contrast


def get_Correlation(p):
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


def get_Energy(p):
    '''
    :param p:P矩阵

    :return energy:能量
    :rtype float
    '''
    energy = 0

    energy = np.sum(p.copy()**2)
    return energy


def get_Homegeneity(p):
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


def get_T(ColorSize,input_img, d):
    '''
    :param ColorSize:颜色量级
    :param d: 距离d

    :return T: theta分别从0到135，4个上述特征的集合
    :rtype list
    '''
    T = []
    for theta in [0, 45, 90, 135]:
        p = get_P_Matrix(ColorSize, input_img, d, theta)
        contrast = get_Contrast(p)
        T.append(contrast)
        correlation = get_Correlation(p)
        T.append(correlation)
        energy = get_Energy(p)
        T.append(energy)
        homegeneity = get_Homegeneity(p)
        T.append(homegeneity)

    return T

# 将图像分成s*s的块 , s = 64


def get_B(input_img, s=64):
    '''
    :param input_img: 输入图像
    :param s: 输入图像分割成好多个块，每个块是s*s的像素

    :return B:分割后并索引好的块矩阵
    :rtype list
    '''
    S = input_img.shape[0]
    row = S//s
    col = row
    N = (S//s)**2
    B = [0]*N
    for i in range(N):
        pos_x = (i//row)*s
        pos_y = (i % row)*s
        B[i] = input_img[pos_x:pos_x+s, pos_y:pos_y+s, 0]
        assert B[i].shape == (64, 64)

    return B


# DCT 变换
def DCT_transformation(img_src):
    '''
    :param img_src:源图像

    :return img_dct:dct变换后的图像
    :rtype ndarray(64*64)
    '''
    img_src = np.float32(img_src)  # 将数值精度调整为32位浮点型
    img_dct = cv2.dct(img_src)  # 使用dct获得img的频域图像
    return img_dct


def get_C_Matrix(N, B):
    '''
    :param N:分割的块数
    :param B:分割后的块矩阵

    :return C:DCT变换后的C矩阵
    :rtype list
    '''
    C = [0]*N
    for i in range(N):
        C[i] = DCT_transformation(B[i])
    return C

# 获取Q向量


def get_Q(C, s, N):
    '''
    :param C: DCT变换后的C矩阵
    :param s: s*s 的小块
    :param N: 分割的块数

    :return Q: C[0,1]～C[0,n],C[1,0]~C[n,0] 组合后的向量
    :rtype ndarray(64*64)
    '''
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


# 获取U向量
def get_U(Q, N):
    '''
    :param Q: C[0,1]～C[0,n],C[1,0]~C[n,0] 组合后的向量
    :param N: 分割后的块数

    :return U: 归一化Q
    :rtype ndarray(64*64)
    '''
    # 归一化Q
    U = np.zeros(Q.shape)
    for i in range(N):
        amax = Q[i, :].max()
        amin = Q[i, :].min()
        U[i] = (Q[i] - amin) / (amax-amin)

    return U


# 计算U向量内部的欧式距离

def caculate_Euclidean_Distance(U):
    '''
    :param U: 归一化的Q

    :return d: U向量内部的欧式距离
    :rtype list
    '''
    # U0是每一行的均值，U的其他不变, U0.shape = (64,1)
    U0 = np.mean(U, axis=1).reshape(U.shape[0], 1)
    assert U0.shape == (64, 1)
    d = [0]*len(U)
    for i in range(len(U)):
        d[i] = np.sqrt(np.sum((U[:, i] - U0)**2))

    return d

# 将T和D 特征向量 组合成H1


def get_H1(T, D):
    '''
    :param T: 16个Global Statistical Feature 特征
    :param D: 64个Local Invariant Feature 特征

    :return H1: [T D]
    :rtype list
    '''
    H1 = [] + T + D
    return H1

# quantize H1 to integer h(q) , hash h = [h(q)] (1<=q<=80)


def quantize(H1):
    '''
    :param H1: [T D]

    :return h: 量化后的H1
    :rtype list
    '''
    H1 = np.array(H1)
    h = np.around(H1*10 + 0.5)
    return list(h)


def get_h(inpath):
    input_img = get_input_img(inpath)
    p = get_P_Matrix(ColorSize, input_img, d, theta)
    contrast = get_Contrast(p)
    correlation = get_Correlation(p)
    energy = get_Energy(p)
    homegeneity = get_Homegeneity(p)
    T = get_T(ColorSize,input_img, d)
    assert len(T) == 16

    B = get_B(input_img, N)
    C = get_C_Matrix(N, B)
    Q = get_Q(C, s, N)
    U = get_U(Q, N)
    D = caculate_Euclidean_Distance(U)
    assert len(D) == 64
    H1 = get_H1(T, D)
    assert len(H1) == 80

    h = quantize(H1)
    return h


# if __name__ == '__main__':
#     inpath = "./DataSet/Standard_benchmark_image/Airplane.tiff"
#     h = get_h(inpath)
#     print('len(h):', len(h))
#     for i in range(0, len(h), 4):
#         print(h[i:i+4])
