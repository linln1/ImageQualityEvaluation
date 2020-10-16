import cv2
import numpy as np
import math



## 无参考情况下的图像清晰度评估方法
def brenner(img):
    '''
    :param img: 2D grey Image
    :return:    a float number that indicates the sharpness of the image
    '''
    #height, width = len(img), len(img[0])
    height, width = np.shape(img)
    sharpness = 0.0
    for x in range(0, height-2):
        for y in range(0, width):
            sharpness += (img[x][y] - img[x+2][y])**2
    return sharpness

def Laplacian(img):
    '''
    :param img: 2D grey Image
    :return:    a float number that indicates the sharpness of the image
    '''
    height, width = np.shape(img)
    assert height == 256 and width == 256
    return cv2.Laplacian(img, cv2.CV_64F).var()

def SMD(img):
    '''
    :param img: narray 2D grey image
    :return: float 图像越清晰越大
    '''
    shape = np.shape(img)
    sharpness = 0.0
    for x in range(1, shape[0]-1):
        for y in range(0, shape[1]):
            sharpness +=math.fabs(img[x,y]-img[x,y-1])
            sharpness +=math.fabs(img[x,y]-img[x+1,y])
    return sharpness

def SMD2(img):
    '''
    :param img:
    :return:
    '''
    shape = np.shape(img)
    sharpness = 0.0
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]-1):
            sharpness += math.fabs(img[x,y]-img[x+1,y])*math.fabs(img[x,y]-img[x,y+1])
    return sharpness

def Vollath(img):
    '''
    :param img:narray 2D gery image
    :return: float number
    '''
    shape = np.shape(img)
    u = np.mean(img)
    sharpness = -shape[0]*shape[1]*(u**2)
    for x in range(0, shape[0]-1):
        for y in range(0, shape[1]):
            sharpness+= img[x,y] * img[x+1,y]
    return sharpness



## 有参考图像情况下的清晰度// (其实这个不光是清晰度了) 就是比较两个图像的差异
def Divergence(path_fake_img, path_real_img):
    '''
    :param path_fake_img: 生成图片的路径
    :param path_real_img: 真实图片的路径
    :return:   返回两张图片的散度损失
    '''
    fake_img = cv2.imread(path_fake_img)
    real_img = cv2.imread(path_real_img)
    assert fake_img.shape[0] == real_img.shape[0] and fake_img.shape[1] == real_img.shape[1]
    D_AB = 0.0
    height, width, channel = fake_img.shape[0], fake_img.shape[1], fake_img.shape[2]
    for i in range(height):
        for j in range(width):
            for k in range(channel):
                D_AB += fake_img[i][j][k] * np.log(fake_img[i][j][k] / real_img[i][j][k]) - fake_img[i][j][k] + real_img[i][j][k]

    return D_AB


def showImage(filepath ,i: int):
    '''
    :param filepath: 存放生成图片和真实图片的路径
    :param i: 第i张图片
    :return:  展示两张图片
    '''
    fake_img = cv2.imread(filepath + str(i) + ".png")
    real_img = cv2.imread(filepath + str(i) + ".png")
    height, width, channel = fake_img.shape[0], fake_img.shape[1], fake_img.shape[2]
    print(height, width, channel)
    concat_img = np.zeros((height, width * 2, channel), np.uint8)
    concat_img[0:height, 0:width, :] = fake_img[0:height, 0:width, :]
    concat_img[0:height, width:, :] = real_img[0:height, 0:width, :]
    cv2.imshow("", concat_img)
    cv2.waitKey(0)

def evaluate(fake, real):
    pass

if __name__ == '__main__':
    for i in range(1000):
        showImage(i)
        img1 = "C:\\linln\\picture\\fake\\" + str(i) + ".png"
        img2 = "C:\\linln\\picture\\real\\" + str(i) + ".png"
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        evaluate(img1,img2)