import tomophantom
from tomophantom import TomoP3D
from scipy.io import savemat
import astra
import matplotlib.pyplot as plt
import numpy as np
import astra_plot
import pylab
import os
import pydicom
from pydicom import dcmread
import math
import cv2

filePath = 'D:/学习/研究生/Code/师姐重建代码/data/2021-10-11-东大seu/2021-11-10/2021-11-10-东大金属seu/重建图/1.2.0.20211110.160055.无金属/16.799/'
# savePath = 'E:/ADN/YuZhou/adn_myrecon/data/nature_image/result6_epoch19/nature_image/art+metal_recon/'
proj_path = 'D:/学习/研究生/Code/师姐重建代码/result/recon2projection/'

name1 = os.listdir(filePath)  # name1中是所有png文件的名称

n = len(name1)
img_data = []
recon_data = []

# for k in range(n):
#     # source = os.path.join(filePath, ' ('+str(k+1)+')'+'.png')
#     source = os.path.join(filePath, str(k) + '.png')
#     temp = cv2.imread(source, 0)
#     # print(type(temp.pixel_array))
#     img_data.append(temp)
#     print(k)
#
# img_data = np.array(img_data)
# # img_data = img_data.transpose(1, 0, 2)

for k in range(n):
    source = os.path.join(filePath, name1[k])
    temp = dcmread(source)
    temp = temp.pixel_array
    #recon_temp = temp
    temp = (temp / 7868)*255;
    #temp = -np.log(temp / 7868)

    # print(type(temp.pixel_array))
    img_data.append(temp)
    #recon_data.append(recon_temp)
    print(k)

img_data = np.array(img_data)
#recon_data = np.array(recon_data)
#img_data = img_data.transpose(1, 0, 2)

print('======================图像数据类型和维度以及最大最小值========================')
print(type(img_data))
print(img_data.shape)
print(type(img_data[1]))
print(img_data.max())
print(img_data.min())
print('======================图像数据类型和维度以及最大最小值========================')

N_size = 200

slice_ = int(0.5*N_size)
plt.gray()
plt.figure(figsize=[30, 30])
plt.subplot(121)
plt.imshow(img_data[slice_, :, :], vmin=0, vmax=255)
plt.title('Axial view')

plt.subplot(122)
plt.imshow(img_data[:, slice_, :], vmin=0, vmax=255)
plt.title('Coronal view')


# =======================接下来是进行投影的部分了=================================

angles = np.arange(0, 2*np.pi, np.pi/360, dtype="float32")   # in radian
angles_deg = np.arange(0, 360, 0.5, dtype="float32")  # in degree

# ----------Distance between two adjacent detector pixels----------
detector_width = 1
detector_height = 1
# -------Number of detector pixels-------------
det_row = 1695
det_col = 1095
# -------------------------
source_origin = 1165.8
origin_det = 388.6
# source_origin = 2000
# origin_det = 4000
a=1200
b=600

# -------------------------

vectors = np.zeros((720, 12), dtype='float')  # 720个投影

for i in range(720):
    # source
    vectors[i, 0] = math.sin(angles[i]) * a
    vectors[i, 1] = -math.cos(angles[i]) * b
    vectors[i, 2] = 0

    # center of detector
    vectors[i, 3] = -math.sin(angles[i]) * a
    vectors[i, 4] = math.cos(angles[i]) * b
    vectors[i, 5] = 0

    # vector from detector pixel(0, 0) to (0, 1)
    vectors[i, 6] = math.cos(angles[i]) * detector_width
    vectors[i, 7] = math.sin(angles[i]) * detector_height
    vectors[i, 8] = 0

    # vector from detector pixel (0, 0) to (1, 0)
    vectors[i, 9] = 0
    vectors[i, 10] = 0
    vectors[i, 11] = detector_height

# for i in range(720):
#     # source
#     vectors[i, 0] = math.sin(angles[i]) * source_origin
#     vectors[i, 1] = -math.cos(angles[i]) * source_origin
#     vectors[i, 2] = 0
#
#     # center of detector
#     vectors[i, 3] = -math.sin(angles[i]) * origin_det
#     vectors[i, 4] = math.cos(angles[i]) * origin_det
#     vectors[i, 5] = 0
#
#     # vector from detector pixel(0, 0) to (0, 1)
#     vectors[i, 6] = math.cos(angles[i]) * detector_width
#     vectors[i, 7] = math.sin(angles[i]) * detector_height
#     vectors[i, 8] = 0
#
#     # vector from detector pixel (0, 0) to (1, 0)
#     vectors[i, 9] = 0
#     vectors[i, 10] = 0
#     vectors[i, 11] = detector_height

    # # vector from detector pixel(0, 0) to (0, 1)
    # vectors[i, 6] = 0
    # vectors[i, 7] = 0
    # vectors[i, 8] = detector_height
    #
    # # vector from detector pixel (0, 0) to (1, 0)
    # vectors[i, 9] = math.cos(angles[i]) * detector_width
    # vectors[i, 10] = math.sin(angles[i]) * detector_height
    # vectors[i, 11] = 0


######## Creating Projection Geometry ##########
######## 创建由 3D 矢量指定的 3D 锥束几何########
# proj_geom = astra.create_proj_geom('cone', detector_width, detector_height, det_row, det_col, angles, source_origin, origin_det)
proj_geom = astra.create_proj_geom('cone_vec',  det_row, det_col, vectors)
######## Creating Volume geometry ##################


#######创建 3D 体积几何体#######
#vol_geo = astra.create_vol_geom(364, 364, 768)
vol_geo = astra.create_vol_geom(364, 364, 565)

#3D模型生成
plt.rcParams['figure.figsize'] = [10, 10]
plot_proj = astra_plot.plot_geom(proj_geom)
astra_plot.plot_geom(vol_geo, plot_proj)

#得到投影结果
#######创建图像 (3D) 的前向投影######
proj_id, proj_data = astra.create_sino3d_gpu(img_data, proj_geom, vol_geo)

print('======================投影数据类型和维度以及最大最小值========================')
print(type(proj_data))
print(proj_data.shape)
print(type(proj_data[1]))
print(proj_data.max())
print(proj_data.min())
print('======================投影数据类型和维度以及最大最小值========================')

###创建一个 3D 对象###
rec_id = astra.data3d.create('-vol', vol_geo)

# 显示投影数据
intens_max = 30154
plt.figure(figsize=[10, 20])
plt.subplot(131)
plt.imshow(proj_data[:, 0, :], vmin=0, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(proj_data[slice_, :, :], vmin=0, vmax=intens_max, aspect="auto")
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(proj_data[:, :, slice_], vmin=0, vmax=intens_max, aspect="auto")
plt.title('Tangentogram view')


# for k in range(720):
#     source = os.path.join(proj_path, str(k+1)+'.mat')
#     # temp = np.squeeze(proj_data[k, :, :], axis=0)
#     savemat(source, {'proj': proj_data[:, k, :]})
#     print(k)

######创建与 ASTRA 工具箱一起使用的 dict#######
cfg = astra.astra_dict('FDK_CUDA')
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id

print(cfg)

# Create the algorithm object from the configuration structure
####创建算法对象###
alg_id = astra.algorithm.create(cfg)

# Run N iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
#######运行算法#####
astra.algorithm.run(alg_id, 200)

# Get the result
rec = astra.data3d.get(rec_id)

print('======================重建数据类型和维度以及最大最小值========================')
print(type(rec))
print(rec.shape)
print(type(rec[1]))
print(rec.max())
print(rec.min())
print('======================重建数据类型和维度以及最大最小值========================')

N_size = 200
slice_ = int(0.5*N_size)

# m1 = rec.max()
# m2 = rec.min()
#
# rec = (rec - m2) / (m1 - m2)
#
# rec = rec * 255

vmax = 255
vmin = 0
plt.gray()
plt.figure(figsize=[30, 30])
plt.subplot(121)
plt.imshow(rec[slice_, :, :], vmin=vmin, vmax=vmax)
plt.title('Axial view')

plt.subplot(122)
plt.imshow(rec[:, slice_, :], vmin=vmin, vmax=vmax)
plt.title('Coronal view')

'''
for k in range(n):
    save_dir = os.path.join(savePath, str(k)+'.png')
    cv2.imwrite(save_dir, rec[k, :, :])
    print(k)
'''

plt.show()

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
######释放内存和显存#######
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)


