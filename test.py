import cv2
import numpy as np

def split(a):  # 获取各行/列起点和终点
    # b是a的非0元素的下标 组成的数组 (np格式),同时也是高度的值
    b = np.transpose(np.nonzero(a))
    star = []
    end = []
    star.append(int(b[0]))
    for i in range(len(b) - 1):
        cha_dic = int(b[i + 1]) - int(b[i])  # 下一个位置跟前一个位置差距
        if cha_dic > 1:  # 下一个位置跟前一个位置差距 大于1,就记录end和下一个star
            # print(cha_dic,int(b[i]),int(b[i+1]))
            end.append(int(b[i]))
            star.append(int(b[i + 1]))
    end.append(int(b[len(b) - 1]))
    return star, end

def get_horizontal_shadow(img, img_bi):  # 水平投影+分割
    # 1.水平投影
    h, w = img_bi.shape
    shadow_h = img_bi.copy()  # shadow_h画图用(切记！copy后面还有个())
    a = [0 for z in range(0, h)]  # 初始化一个长度为h的数组，用于记录每一行的黑点个数
    for j in range(0, h):  # 遍历一行
        for i in range(0, w):  # 遍历一列
            if shadow_h[j, i] == 255:  # 发现白色
                a[j] += 1  # a数组这一行的值+1
                shadow_h[j, i] = 0  # 记录好了就变为黑色
    for j in range(0, h):  # 遍历一行 画黑条,长度为a[j]
        for i in range(0, a[j]):
            shadow_h[j, i] = 0
    # 2.开始分割
    # step2.1: 获取各行起点和终点
    star, end = split(a)
    cropped_images = []  # 用于存储切割后的图像
    # step2.2: 切割[y:y+h, x:x+w]
    for l in range(len(star)):  # 就是几行
        ys = star[l]
        ye = end[l]
        if ye-ys>40:
            img_crop = img_bi[ys:ye, 0:w]
            # cv2.imwrite(save_path+'\\img_crop_' + str(l) + '.jpg', img_crop)
            cropped_images.append(img_crop)
# 返回两个值，两个值都是图片
    return cropped_images

def find_end(start, arg, black, white, width, black_max, white_max):
    end = start + 1
    for m in range(start + 1, width - 1):
        if (black[m] if arg else white[m]) > (0.95*black_max if arg else 0.95*white_max):
            end = m
            break
    return end

def char_segmentation(thresh):
    """ 分割字符 """
    white, black = [], []    # list记录每一列的黑/白色像素总和
    height, width = thresh.shape
    white_max = 0    # 仅保存每列，取列中白色最多的像素总数
    black_max = 0    # 仅保存每列，取列中黑色最多的像素总数
    # 计算每一列的黑白像素总和
    for i in range(width):
        line_white = 0    # 这一列白色总数
        line_black = 0    # 这一列黑色总数
        for j in range(height):
            if thresh[j][i] == 255:
                line_white += 1
            if thresh[j][i] == 0:
                line_black += 1
        white_max = max(white_max, line_white)
        black_max = max(black_max, line_black)
        white.append(line_white)
        black.append(line_black)
    # arg为true表示黑底白字，False为白底黑字
    arg = True
    if black_max < white_max:
        arg = False
    # 分割车牌字符
    n = 1
    while n < width - 20:
        n += 1
        # 判断是白底黑字还是黑底白字  0.05参数对应上面的0.95 可作调整
        if (white[n] if arg else black[n]) > (0.05 * white_max if arg else 0.05 * black_max):
            start = n
            end = find_end(start, arg, black, white, width, black_max, white_max)
            n = end
            if end - start > 10 or end > (width * 3 / 7):
                cropImg = thresh[0:height, start-1:end+1]
                # 对分割出的数字、字母进行resize并保存
                cropImg = cv2.resize(cropImg, (34, 56))
                cv2.imwrite(save_path + '\\{}.bmp'.format(n), cropImg)
                # cv2.imshow('Char_{}'.format(n), cropImg)
def resize_img(img):
    """ resize图像 """
    h, w = img.shape[:-1]
    scale = 350 / max(h, w)
    img_resized = cv2.resize(img, None, fx=scale, fy=scale,interpolation=cv2.INTER_CUBIC)
    # print(img_resized.shape)
    return img_resized

def shift(img):
    img=resize_img(img)
    border_size=10
    h,w,_=img.shape
    img=img[border_size:h-border_size,:]
    img=img[:, border_size:w-border_size]
    img=resize_img(img)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转为灰度图像
    retval,dst=cv2.threshold(gray,110,255,cv2.THRESH_BINARY)#阈值分割
    #如果要考虑黄底黑字的话就要做个判断黑色和白色的色素块哪个多
    white=len(dst.astype(np.int8)[dst==255])
    black=len(dst.astype(np.int8)[dst==0])
    if black<white:
        retval,dst=cv2.threshold(gray,110,255,cv2.THRESH_BINARY_INV)#阈值分割
    re_img=get_horizontal_shadow(img,dst)
    #膨胀运算
    # 设置卷积核
    kernel = np.ones((10, 10), np.uint8)
    # 图像膨胀处理
    result= cv2.dilate(re_img[0], kernel,1)
    # 查找轮廓
    contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours=contours[::-1]
    # 依次遍历轮廓并切割字符
    #切割汉字
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(re_img[0], (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_char = re_img[0][y:y + h, x:x + w]
        # cv2.imshow('Cropped Character', cropped_char)
        cropImg = cv2.resize(cropped_char, (45, 56))
        cv2.imwrite(save_path + '\\s' + str(contours.index(contour)) + '.jpg', cropImg)
        cv2.waitKey(0)
    char_segmentation(re_img[1])


save_path='D:\\Python practice\\NewLearning\\char_jpg'
img=cv2.imread('fei1.png')
shift(img)

cv2.waitKey(0)
cv2.destroyAllWindows()