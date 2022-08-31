%% 说明
% 该程序用于计算投影图像中的圆心
clc;
clear;
close all;
%cd 'E:\校正程序1\dcmimage_1'
cd 'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\八球\8\dcmimage'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ProjNum = 360;
nBallNum = 8;
ProjPt = zeros(2*nBallNum,ProjNum);

for n = 1:ProjNum
    filename = sprintf('%.3d.dcm',n);
    I = dicomread(filename);
    
    I2 = mat2gray(double(I));
    %figure;imshow(I2,[]);

    I3 = imbinarize(I,0.5);
    I3 = ~I3;
    %figure;imshow(I3,[]);

    Ibw = imfill(I3,'holes');
    %figure;imshow(Ibw,[]);
    %Ibw = bwareaopen(Ibw,100); %该函数用于移除小区域
    %figure; imshow(Ibw,[]);
    hold on;
    [L,m] = bwlabel(Ibw,8);
    stats = regionprops(L,'Centroid');
    Pt = zeros(2,8);
    Pt_r = zeros(2,8);
    for i=1:m
        Pt(1,i) = stats(i).Centroid(1);
        Pt(2,i) = stats(i).Centroid(2);
        %plot(stats(i).Centroid(1),stats(i).Centroid(2),'R+');
    end
    
    [y_sort,sign] = sort(Pt(2,:),'descend');

    Pt_r(2,:) = y_sort;

    for j = 1:m
        Pt_r(1,j) = Pt(1,sign(j));
    end
    ProjPt(:,n) = reshape(Pt_r,[],1);
end

ProjPt2Save = reshape(ProjPt,[],1);
cd ../
save('ProjPt2Save.mat', 'ProjPt2Save')
filePt = 'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\八球\8\pt_irregular_X1Y2Z3_510_516_8_simulation.bin';%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileId = fopen(filePt,'w');
fwrite(fileId,ProjPt2Save,'double');
fclose(fileId);



 %% 备用代码
% 
%  for i = 2:9
% %         index_1 = (i-1)*100+1;
% %         index_2 = i*100;
%         index_x = [101 201 301 401 501 651 751 881];
%         index_y = [200 300 400 500 650 750 880 1000];
%         dis = index_y-index_x+1;
%         space = index_y-100;
% 
% %         g0 = I3(index_1:index_2,:);
%         g0 = I3(index_x(i-1):index_y(i-1),:);
% 
% 
%         s=1:1024;
%         %t=1:1024;
%         t=1:dis(i-1);
%         [x0,y0]=meshgrid(s,t);
%         x=sum(x0.*g0)/sum(g0);
%         y=sum(y0.*g0)/sum(g0);
% %         figure,imshow(g0);hold on;
% %         plot(x,y,'o','LineWidth',4);
%         if i==2 
%             y = y+100;
%         else
%             y = y+index_y(i-2);
%         end
%         j = abs((i-1)-9); %第一个球在最下面，存的时候1-8调换一下位置
%         ProjPt(j*2-1,n) = x;
%         ProjPt(j*2,n) = y;
% end
% 



%% 备用代码
% s = regionprops(I2,'centroid');
% centroids = cat(1,s.Centroid);

%g0=mat2gray(double(I));
%ProjectPt_vec2=reshape(ProjectPt_vec2,[],1);