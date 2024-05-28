# 不要重复执行,图像结果新路径载入会造成错误,后续完善
import cv2 as cv  # cv默认采用BGR,但是在PIL等库中多采用RGB,通常在cv操作前后需要用cv.cvtColor(src, cv.COLOR_RGB2BGR, dst)  BGR2RGB BGR2GREY BGR2HSV
import numpy as np  # ndarray  npy/npz
import os  # os.getcwd()

# 图像和标签存储的目录
dataset_load_dir = './resource/CCD/'  # 可考虑递归处理
image_read_dir = './resource/CCD/image/'
# 结果存储路径
image_save_dir = './result/CCD/image/'
label_save_dir = './result/CCD/'
if not os.path.isdir(image_save_dir):
    os.makedirs(image_save_dir)  # 创建多级目录,用于保存结果
# 依序获取图像列表
images = sorted(os.listdir(image_read_dir))
# 读取标签数据npy格式
labels = np.load(dataset_load_dir + 'labels568.npy')  # labels[0]  os.path.join与+
# print(labels.shape)  # 568 3  [[,,]/r/n[,,]...[,,]]

# 预处理图像  (0)缩放(宽高一般无需等比例,等比例所用填充色是噪音) 翻转/旋转/切割/色彩转换-扩充等
for index, image in enumerate(images, start=1):
    if image.endswith('.png'):
        # 读取图像 ('图像路径', 读取方式参数)  1:彩色模式加载,忽略透明度,默认参数; 0:灰度模式加载; -1:包括alpha通道的彩色加载
        image0 = cv.imread(os.path.join(image_read_dir, image), 1)  # cv-channel:BGR  显示读取的图像:matplotlib.pyplot plt.imshow(img[:,:,:,-1]) 翻转bgr
        # 缩放 resize(img矩阵, (w, h), 插值方式)
        image1 = cv.resize(image0, (224, 224), interpolation=cv.INTER_LINEAR)  # 后续模型训练和推理采用的插值方式和次数最好保持一致
        # INTER_NEAREST/INTER_LINEAR:双线性插值(默认)/INTER_CUBIC:4x4像素邻域的双三次插值/INTER_AREA:像素区域关系重采样/INTER_LANCZOS4:8x8像素邻域的Lanczos插值
        cv.imwrite(os.path.join(image_save_dir, image), image1)  # 写入保存图像结果 ('已存在的路径/fileFullName', 图像变量)  同名会覆盖;图像恢复RGB
        # 缩放对色彩标签数据影响忽略不计,光源值在插值计算中不能算丢失,而是被整合了
new_labels = labels

# 取第一张图和第七张图检验一下结果
img_res = cv.imread(os.path.join(image_save_dir, images[0]))  # image.shape [:2]
label_res = new_labels[0]
print(label_res, labels[0], sep='\n', end='\n')
cv.imshow('0*1', img_res)  # ('窗口名称', 图像变量) esc
cv.waitKey(0)  # 在imshow执行之后调用waitKey给绘制图像预留时间,防止无响应或显示失败 anyKey
img_res = cv.imread(os.path.join(image_save_dir, images[6]))
label_res = new_labels[6]
print(label_res, labels[6], sep='\n', end='\n')
cv.imshow('0*7', img_res)
cv.waitKey(0)
cv.destroyAllWindows()  # 销毁窗口
print('缩放效果检验结束!')

new_images = sorted(os.listdir(image_save_dir))
# 扩充 选择直角旋转和翻转两种方式,对原始图像的色彩和标签结果影响较小
# (1)直角旋转270
for index, image in enumerate(new_images, start=1):
    if image.endswith('.png'):
        image0 = cv.imread(os.path.join(image_save_dir, image))
        image1 = cv.rotate(image0, cv.ROTATE_90_COUNTERCLOCKWISE)  # ~顺时针270  90:ROTATE_90_CLOCKWISE 180:ROTATE_180
        new_image = f'1{index:04d}.png'  # 更改名称,首位数字用于标记预处理方式
        cv.imwrite(os.path.join(image_save_dir, new_image), image1)
        # 标签数据需要同步扩展,标签值不用改变
        new_labels = np.concatenate([new_labels, [new_labels[index - 1]]], axis=0)  # 维度:0行1列2高  vstack/stack/hstack/dstack append

# (2)水平翻转
for index, image in enumerate(new_images, start=1):
    if image.endswith('.png'):
        image0 = cv.imread(os.path.join(image_save_dir, image))
        image1 = cv.flip(image0, 1)  # 1:水平翻转 0:垂直翻转 -1:水平垂直翻转~旋转pi
        new_image = f'2{index:04d}.png'  # 更改名称,首位数字用于标记预处理方式
        cv.imwrite(os.path.join(image_save_dir, new_image), image1)
        # 标签数据需要同步扩展,标签值不用改变
        new_labels = np.concatenate([new_labels, [new_labels[index - 1]]], axis=0)

# 保存修改后的Numpy标签数组
np.save(label_save_dir + 'labels568_ext012.npy', new_labels)


# 取3组图检验一下结果
label_res = np.load(label_save_dir + 'labels568_ext012.npy')
print(labels[0], label_res[0], label_res[568], label_res[1136], sep='\n', end='\n----------\n')
print(labels[6], label_res[6], label_res[574], label_res[1142], sep='\n', end='\n----------\n')
print(labels[567], label_res[567], label_res[1135], label_res[1703], sep='\n', end='\n----------\n')
new_images = sorted(os.listdir(image_save_dir))
img_res = cv.imread(os.path.join(image_save_dir, new_images[0]))
cv.imshow('0*1', img_res)  # ('窗口名称', 图像变量) esc
cv.waitKey(0)  # 在imshow执行之后调用waitKey给绘制图像预留时间,防止无响应或显示失败 anyKey
img_res = cv.imread(os.path.join(image_save_dir, new_images[568]))
cv.imshow('1*1', img_res)
cv.waitKey(0)
img_res = cv.imread(os.path.join(image_save_dir, new_images[1136]))
cv.imshow('2*1', img_res)
cv.waitKey(0)
cv.destroyAllWindows()  # 销毁窗口

img_res = cv.imread(os.path.join(image_save_dir, new_images[6]))
cv.imshow('0*7', img_res)
cv.waitKey(0)
img_res = cv.imread(os.path.join(image_save_dir, new_images[574]))
cv.imshow('1*7', img_res)
cv.waitKey(0)
img_res = cv.imread(os.path.join(image_save_dir, new_images[1142]))
cv.imshow('2*7', img_res)
cv.waitKey(0)
cv.destroyAllWindows()  # 销毁窗口

img_res = cv.imread(os.path.join(image_save_dir, new_images[567]))
cv.imshow('0*568', img_res)
cv.waitKey(0)
img_res = cv.imread(os.path.join(image_save_dir, new_images[1135]))
cv.imshow('1*568', img_res)
cv.waitKey(0)
img_res = cv.imread(os.path.join(image_save_dir, new_images[1703]))
cv.imshow('2*568', img_res)
cv.waitKey(0)
cv.destroyAllWindows()  # 销毁窗口
