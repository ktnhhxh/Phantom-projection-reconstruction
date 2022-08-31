%寻找文件夹中dcm图片最大值
clc;
clear all;
file_path = 'D:\学习\研究生\Code\50um原图+去噪\50um\50um\denoise109\640img\';
img_path_list = dir(strcat(file_path,'*.dcm'));
img_num = length(img_path_list);
img_max=0;
img_min=10000;
if img_num > 0 %有满足条件的图像
    for num = 1:img_num %逐一读取图像
        num
        image_name = img_path_list(num).name;
%         A =dicomread(strcat(file_path,image_name));
        A =imread(strcat(file_path,image_name));
        temp=max(max(A));
        temp1=min(min(A));
        if temp>img_max
            img_max=temp;
        end
        if temp1<img_min
            img_min=temp1;
        end
    end
end
