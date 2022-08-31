clc;
clear;
I=dicomread('D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\球img\ballprojdata1\001.dcm');
%I=dicomread('D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\球img\ballprojdata\001.dcm');
%比如“E:\image\101.bmp”
g0=mat2gray(double(I));
s=1:1024;
t=1:1024;
[x0 y0]=meshgrid(s,t);
I=x0.*g0;
x=sum(x0.*g0)/sum(g0);
y=sum(y0.*g0)/sum(g0);
figure,imshow(g0);hold on;%保留当前图像和坐标
plot(x,y,'o','LineWidth',4);
x,y