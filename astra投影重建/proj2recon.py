import tomophantom
from tomophantom import TomoP3D
import astra
from scipy.io import savemat
import matplotlib.pyplot as plt
import numpy as np
import astra_plot
import scipy.misc
import pylab
import os
import pydicom
from pydicom import dcmread
import math
import cv2
import SimpleITK as sitk

filePath = 'D:/学习/研究生/Code/师姐重建代码/老师投影重建代码/data/圆柱球img/圆柱球结果/cylinderballincline_projdata/'
#filePath = 'D:/学习/研究生/Code/师姐重建代码/老师投影重建代码/data/圆柱球img/圆柱球结果/ellipse_cylinderballprojdata'
save_path = 'D:/学习/研究生/Code/师姐重建代码/老师投影重建代码/data/圆柱体+球/圆柱球1/reconstructincline/'
#save_path = 'D:/学习/研究生/Code/师姐重建代码/老师投影重建代码/data/圆柱球img/圆柱球结果/ellipse_reconstruct'
name1 = os.listdir(filePath)  # name1中是所有dcm文件的名称
n = len(name1)
proj_data = []
print(name1)


for k in range(n):
    source = os.path.join(filePath, name1[k])
    temp = dcmread(source)
    temp = temp.pixel_array
    #temp = 32768-temp
    #temp = (temp / 65535) * 255;
    temp = -np.log(temp+1/63500)

    # print(type(temp.pixel_array))
    proj_data.append(temp)
    print(k)

proj_data = np.array(proj_data)
proj_data = proj_data.transpose(1, 0, 2)

print('======================投影数据类型和维度以及最大最小值========================')
print(type(proj_data))
print(proj_data.shape)
print(type(proj_data[1]))
print(proj_data.shape)
print(proj_data.max())
print(proj_data.min())
print('======================投影数据类型和维度以及最大最小值========================')

slice_ = 400

# 显示投影数据
plt.gray()
intens_max = proj_data.max()
intens_min = proj_data.min()
plt.figure(figsize=[10, 20])
plt.subplot(131)
plt.imshow(proj_data[:, 0, :], vmin=intens_min, vmax=intens_max)
plt.title('2D Projection (analytical)')
plt.subplot(132)
plt.imshow(proj_data[slice_, :, :], vmin=intens_min, vmax=intens_max, aspect = "auto")
plt.title('Sinogram view')
plt.subplot(133)
plt.imshow(proj_data[:, :, slice_], vmin=intens_min, vmax=intens_max, aspect = "auto")
plt.title('Tangentogram view')

# plt.show()

#=============================================================

angles = np.arange(0, 2*np.pi, np.pi/180, dtype="float32")   # in radian
angles_deg = np.arange(0, 360, 0.5, dtype="float32")  # in degree

#----------Distance between two adjacent detector pixels----------
detector_width = 1
detector_height = 1
#-------Number of detector pixels-------------
det_row = 1024
det_col = 1024
#-------------------------
# source_origin = 1167
# origin_det    = 389

source_origin = 4000
origin_det    = 4000
a=4000
b=2000
#-------------------------
vectors = np.zeros((360, 12), dtype='float')  # 720个投影

for i in range(360):
    # source
    vectors[i, 0] = math.sin(angles[i]) * source_origin
    vectors[i, 1] = -math.cos(angles[i]) * source_origin
    vectors[i, 2] = 0

    # center of detector
    vectors[i, 3] = -math.sin(angles[i]) * origin_det
    vectors[i, 4] = math.cos(angles[i]) * origin_det
    vectors[i, 5] = 0

    # vector from detector pixel(0, 0) to (0, 1)
    vectors[i, 6] = math.cos(angles[i]) * detector_width
    vectors[i, 7] = math.sin(angles[i]) * detector_height
    vectors[i, 8] = 0

    # vector from detector pixel (0, 0) to (1, 0)
    vectors[i, 9] = 0
    vectors[i, 10] = 0
    vectors[i, 11] = detector_height


# for i in range(360):
#     # source
#     vectors[i, 0] = math.sin(angles[i]) * a
#     vectors[i, 1] = -math.cos(angles[i]) * b
#     vectors[i, 2] = 0
#
#     # center of detector
#     vectors[i, 3] = -math.sin(angles[i]) * a
#     vectors[i, 4] = math.cos(angles[i]) * b
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



######## Creating Volume geometry ##################
vol_geo = astra.create_vol_geom(1024, 1024, 1024)
rec_id = astra.data3d.create('-vol', vol_geo)
print(rec_id)

# proj_geom = astra.create_proj_geom('cone', detector_width, detector_height, det_row, det_col, angles, source_origin, origin_det)
proj_geom = astra.create_proj_geom('cone_vec',  det_row, det_col, vectors)

plt.rcParams['figure.figsize'] = [10, 10]
plot_proj = astra_plot.plot_geom(proj_geom)
astra_plot.plot_geom(vol_geo, plot_proj)

proj_id = astra.data3d.create('-sino', proj_geom, proj_data)



print(proj_id)
print("投影id")
print(astra.data3d.get(proj_id).shape)

cfg = astra.astra_dict('FDK_CUDA')

cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cfg)




# Run 150 iterations of the algorithm
# Note that this requires about 750MB of GPU memory, and has a runtime
# in the order of 10 seconds.
astra.algorithm.run(alg_id, 200)

# Get the result
rec = astra.data3d.get(rec_id)

print('======================重建图像数据类型和维度以及最大最小值========================')
print(type(rec))
print(rec.shape)
print(rec[0, :, :].shape)
print(rec.max())
print(rec.min())
print('======================重建图像数据类型和维度以及最大最小值========================')


print('保存最后结果')
for i in range(1024):
    save_dir = os.path.join(save_path, str(i)+'.mat')
    temp = rec[i, :, :]
    savemat(save_dir, {'recon': temp})
    print(i)

# for k in range(n):
#     save_dir = os.path.join(save_path, str(k)+'.png')
#     temp = rec[i, :, :]
#     scipy.misc.imsave('D:/学习/研究生/Code/师姐重建代码/老师投影重建代码/data/圆柱体+球/圆柱球1/reconstruct/png/%d.jpg' % k, temp)
#     print(k)


N_size = 328
slice_ = int(N_size)


max1 = rec.max()
min1 = rec.min()

# rec = (rec - min1) / (max1 - min1)
#
# rec = rec * 65535-32767

vmax = max1
vmin = min1
plt.gray()
plt.figure(figsize = [20,30])
plt.subplot(121)
plt.imshow(rec[slice_,:,:], vmin=vmin, vmax=vmax)
plt.title('Axial view')

plt.subplot(122)
plt.imshow(rec[:,slice_,:], vmin=vmin, vmax=vmax)
plt.title('Coronal view')

plt.show()

# Clean up. Note that GPU memory is tied up in the algorithm object,
# and main RAM in the data objects.
astra.algorithm.delete(alg_id)
astra.data3d.delete(rec_id)
astra.data3d.delete(proj_id)

