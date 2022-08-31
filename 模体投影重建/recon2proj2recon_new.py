import tomophantom
from tomophantom import TomoP3D
from scipy.io import savemat
from scipy.io import loadmat
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
import struct

# 重建图所在位置
filePath = 'E:/SimulateProjImage/Cylinder/CylinderImage_Inside_0.01_10_final/'
# 投影图保存位置
proj_path = 'E:/SimulateProjImage/ProjDcm/cylinderballprojdata_final/'
proj_path_rec = 'E:/SimulateProjImage/ProjDcm/cylinderballprojdata_final_rec/'  # 结果参数得到的投影图
recon_path = 'E:/SimulateProjImage/Recon/'
# 文件读取
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
# img_data = img_data.transpose(1, 0, 2)

for k in range(n):
    source = os.path.join(filePath, name1[k])
    temp = dcmread(source)
    temp = temp.pixel_array
    # recon_temp = temp
    # 进行标准化
    temp = (temp / 65535)*255
    # print(type(temp.pixel_array))
    img_data.append(temp)
    # recon_data.append(recon_temp)
    print(k)

img_data = np.array(img_data)
# recon_data = np.array(recon_data)
# img_data = img_data.transpose(1, 0, 2)

print('======================图像数据类型和维度以及最大最小值========================')
print(type(img_data))
print(img_data.shape)
print(type(img_data[1]))
print(img_data.max())
print(img_data.min())
print('======================图像数据类型和维度以及最大最小值========================')

# N_size = 400
# slice_ = int(0.5*N_size)
# plt.gray()
# plt.figure(figsize=[30, 30])
# plt.subplot(121)
# # plt.savefig('2000-4000recon_.Axial-view.jpg')
# plt.imshow(img_data[slice_, :, :], vmin=0, vmax=255)
# plt.title('Axial view')
#
# plt.subplot(122)
# # plt.savefig('2000-4000recon_.Coronal-view.jpg')
# plt.imshow(img_data[:, slice_, :], vmin=0, vmax=255)
# plt.title('Coronal view')


# =======================接下来是进行投影的部分了=================================

angles = np.arange(0, 2*np.pi, np.pi/180, dtype="float32")   # in radian
angles_deg = np.arange(0, 360, 0.5, dtype="float32")  # in degree

# -------Number of detector pixels-------------
det_row = 1344
det_col = 2016
det_row_half = det_row/2
det_col_half = det_col/2
det_center = np.zeros((3, 1), dtype='float')
det_center[0] = 0
det_center[1] = det_col_half
det_center[2] = det_row_half
u = np.zeros((3, 1), dtype='float')
u[1] = 1
v = np.zeros((3, 1), dtype='float')
v[2] = 1

# -------------------------投影vectors设定 begin ------------------------- #
vectors = np.zeros((360, 12), dtype='float')  # 360个角度
sourceD = np.zeros((3, 1), dtype='float')
RotateMatrix = np.zeros((3, 3), dtype='float')  # 探测器坐标系转到物体坐标系的旋转矩阵
trans = np.zeros((3, 1), dtype='float')

sourceD[0] = 8000
sourceD[1] = 1000
sourceD[2] = 650
a = 4000
b = 3900
trans[1] = 1000
trans[2] = 650
thetaX = 1/180*np.pi  # 我们的坐标系中倾角
thetaY = 2/180*np.pi

for i in range(360):
    thetaZ = 3/180*np.pi + angles[i]

    RotateMatrix[0][0] = math.cos(thetaY)*math.cos(thetaZ)
    RotateMatrix[0][1] = math.cos(thetaY)*math.sin(thetaZ)
    RotateMatrix[0][2] = math.sin(thetaY)
    RotateMatrix[1][0] = -math.sin(thetaX)*math.sin(thetaY)*math.cos(thetaZ) - math.cos(thetaX)*math.sin(thetaZ)
    RotateMatrix[1][1] = -math.sin(thetaX)*math.sin(thetaY)*math.sin(thetaZ) + math.cos(thetaX)*math.cos(thetaZ)
    RotateMatrix[1][2] = math.sin(thetaX)*math.cos(thetaY)
    RotateMatrix[2][0] = -math.cos(thetaX)*math.sin(thetaY)*math.cos(thetaZ) + math.sin(thetaX)*math.sin(thetaZ)
    RotateMatrix[2][1] = -math.cos(thetaX)*math.sin(thetaY)*math.sin(thetaZ) - math.sin(thetaX)*math.cos(thetaZ)
    RotateMatrix[2][2] = math.cos(thetaX)*math.cos(thetaY)

    trans[0] = math.sqrt(a*a*math.cos(angles[i])*math.cos(angles[i]) + b*b*math.sin(angles[i])*math.sin(angles[i]))
    sourceX = np.dot(RotateMatrix, (sourceD - trans))
    det_centerX = np.dot(RotateMatrix, (det_center - trans))
    uX = np.dot(RotateMatrix, u)
    vX = np.dot(RotateMatrix, v)
    # source
    vectors[i, 0] = sourceX[1]
    vectors[i, 1] = -sourceX[0]
    vectors[i, 2] = sourceX[2]

    # center of detector
    vectors[i, 3] = det_centerX[1]
    vectors[i, 4] = -det_centerX[0]
    vectors[i, 5] = det_centerX[2]

    # vector from detector pixel(0, 0) to (1, 0)  u
    vectors[i, 6] = uX[1]
    vectors[i, 7] = -uX[0]
    vectors[i, 8] = uX[2]

    # vector from detector pixel (0, 0) to (0, 1)  v
    vectors[i, 9] = vX[1]
    vectors[i, 10] = -vX[0]
    vectors[i, 11] = vX[2]
# -------------------------投影vectors设定  end  ------------------------- #

# ##########################生成投影数据，显示，保存 begin########################## #
proj_geom = astra.create_proj_geom('cone_vec',  det_row, det_col, vectors)
vol_geo = astra.create_vol_geom(1000, 1000, 700)
proj_id, proj_data = astra.create_sino3d_gpu(img_data, proj_geom, vol_geo, True)
# data = astra.data3d.create('-sino', proj_geom)
rec_id_std, rec_std = astra.create_backprojection3d_gpu(proj_id, proj_geom, vol_geo, True)
# rec_id_std, rec_std = astra.create_backprojection3d_gpu(proj_id, proj_geom, vol_geo, True)
plt.gray()
plt.figure(figsize=[20, 20])
plt.subplot(121)
plt.imshow(rec_std[300, :, :])
plt.title('Axial view Ball4')
plt.subplot(122)
plt.imshow(rec_std[60, :, :])
plt.title('Axial view Ball1')
# # 3D模型生成
# plt.rcParams['figure.figsize'] = [10, 10]
# plot_proj = astra_plot.plot_geom(proj_geom)
# astra_plot.plot_geom(vol_geo, plot_proj)
#
# # 显示投影数据
# intens_max = 30154
# plt.figure(figsize=[10, 20])
# plt.subplot(131)
# plt.imshow(proj_data[:, 0, :], vmin=0, vmax=intens_max)
# plt.title('2D Projection (analytical)')

# 存储投影图
# for k in range(360):
#     source = os.path.join(proj_path, str(k+1)+'.mat')
#     # temp = np.squeeze(proj_data[k, :, :], axis=0)
#     savemat(source, {'proj': proj_data[:, k, :]})
#     print(k)
# ##########################生成投影数据，显示，保存   end########################## #

# rec_id, rec = astra.create_backprojection3d_gpu(proj_id, proj_geom, vol_geo, True)
# plt.gray()
# plt.figure(figsize=[30, 30])
# plt.subplot(121)
# plt.imshow(rec[300, :, :])
# plt.title('Axial view')

# -------------------------重建vectors设定 begin ------------------------- #
vectors = np.zeros((360, 12), dtype='float')  # 720个投影
sourceD = np.zeros((3, 1), dtype='float')
trans = np.zeros((3, 1), dtype='float')
RotateMatrix = np.zeros((3, 3), dtype='float')  # 探测器坐标系转到物体坐标系的旋转矩阵

filename = 'reco_SA.mat'
data = loadmat(filename)
vec = np.reshape(data["A"], (360, 10))

for i in range(360):

    sourceD[0] = vec[i][6]
    sourceD[1] = vec[i][7]
    sourceD[2] = vec[i][8]

    trans[0] = vec[i][3]
    trans[1] = vec[i][4]
    trans[2] = vec[i][5]

    thetaX = vec[i][0] / 180 * np.pi  # 我们的坐标系中倾角
    thetaY = vec[i][1] / 180 * np.pi
    thetaZ = vec[i][2] / 180 * np.pi

    RotateMatrix[0][0] = math.cos(thetaY) * math.cos(thetaZ)
    RotateMatrix[0][1] = math.cos(thetaY) * math.sin(thetaZ)
    RotateMatrix[0][2] = math.sin(thetaY)
    RotateMatrix[1][0] = -math.sin(thetaX) * math.sin(thetaY) * math.cos(thetaZ) - math.cos(thetaX) * math.sin(thetaZ)
    RotateMatrix[1][1] = -math.sin(thetaX) * math.sin(thetaY) * math.sin(thetaZ) + math.cos(thetaX) * math.cos(thetaZ)
    RotateMatrix[1][2] = math.sin(thetaX) * math.cos(thetaY)
    RotateMatrix[2][0] = -math.cos(thetaX) * math.sin(thetaY) * math.cos(thetaZ) + math.sin(thetaX) * math.sin(thetaZ)
    RotateMatrix[2][1] = -math.cos(thetaX) * math.sin(thetaY) * math.sin(thetaZ) - math.sin(thetaX) * math.cos(thetaZ)
    RotateMatrix[2][2] = math.cos(thetaX) * math.cos(thetaY)

    sourceX = np.dot(RotateMatrix, (sourceD - trans))
    det_centerX = np.dot(RotateMatrix, (det_center - trans))
    uX = np.dot(RotateMatrix, u)
    vX = np.dot(RotateMatrix, v)
    # source
    vectors[i, 0] = sourceX[1]
    vectors[i, 1] = -sourceX[0]
    vectors[i, 2] = sourceX[2]

    # center of detector
    vectors[i, 3] = det_centerX[1]
    vectors[i, 4] = -det_centerX[0]
    vectors[i, 5] = det_centerX[2]

    # vector from detector pixel(0, 0) to (1, 0)  u
    vectors[i, 6] = uX[1]
    vectors[i, 7] = -uX[0]
    vectors[i, 8] = uX[2]

    # vector from detector pixel (0, 0) to (0, 1)  v
    vectors[i, 9] = vX[1]
    vectors[i, 10] = -vX[0]
    vectors[i, 11] = vX[2]
# -------------------------重建vectors设定   end ------------------------- #

# ##########################生成投影数据，显示，保存 begin  ########################## #
# vol_geo = astra.create_vol_geom(1000, 1000, 700)
proj_geom_rec = astra.create_proj_geom('cone_vec',  det_row, det_col, vectors)
# proj_id_rec, proj_data_rec = astra.create_sino3d_gpu(img_data, proj_geom_rec, vol_geo)

# 3D模型生成
# plt.rcParams['figure.figsize'] = [10, 10]
# plot_proj_rec = astra_plot.plot_geom(proj_geom_rec)
# astra_plot.plot_geom(vol_geo, plot_proj_rec)
# ##########################生成投影数据，显示，保存 end  ########################## #

# 重建
rec_id, rec = astra.create_backprojection3d_gpu(proj_id, proj_geom_rec, vol_geo, True)
plt.gray()
plt.figure(figsize=[20, 20])
plt.subplot(121)
plt.imshow(rec[300, :, :])
plt.title('Axial view Ball4')
plt.subplot(122)
plt.imshow(rec[60, :, :])
plt.title('Axial view Ball1')
# plt.gray()
# plt.figure(figsize=[30, 30])
# plt.subplot(121)
# plt.imshow(rec[60, :, :])
# plt.title('Axial view')
#
# plt.subplot(122)
# plt.imshow(rec[:, 500, :])
# plt.title('Coronal view')

# 存储重建的700层图像
for k in range(700):
    source = os.path.join(recon_path, str(k+1)+'.mat')
    savemat(source, {'recon': rec[k, :, :]})
    print(k)

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
# #####释放内存和显存#######
# astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)
