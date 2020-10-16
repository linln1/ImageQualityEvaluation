# 图像质量评估算法 ::O::

## 首先是基于灰度图的清晰度评估方法
## 如果是彩色图，可以将多通道的值转换成灰度分别进行评估，然后取平均
## 初步考虑使用有餐考和无参考的方法加权
## 其中无参考又结合多种无参考方法，分别对real_img, fake_img进行评估，然后取两者的差值绝对值，最后取平均， 多种无参考方法可以加权

### [To do List]
    [x] No reference Method
    [x] Compare to the Ground truth pic
    [] Other efficient methods to evaluate the sharpness of the generated pics
    

### [Implementation]
    Using::
        - python 3.7.7
        - packages
            - opencv_python(cv2)
            - numpy
            - math