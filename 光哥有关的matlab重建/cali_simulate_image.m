% % cd 'C:\liguang\calibration\calibration&recon_seu\Bin\Config\'
% %   filename   = '2018_07_12_simulation_500.bin';
%     filename = 'C:\liguang\calibration\2018_07_12_medi_phys_ball_radius_8pixel_m300.bin';
% %   fileID = fopen(filename,'w');
% %   fwrite(fileID,ProjectPt2,'double');
% %   fclose(fileID);
%   noise = 1.0*randn(2,projnum);
%   
%   
%   fileID = fopen(filename,'r');
%   A = fread(fileID,720,'double');
%   fclose(fileID);
%   
% %   fileID = fopen(filename_1,'r');
% %   B = fread(fileID,720,'double');
% %   fclose(fileID);
%    A(1:2:720)=A(1:2:720)+noise(1,:)';
%    A(2:2:720)=A(2:2:720)+noise(2,:)';
   
  %% 
%    savefile = 'C:\liguang\calibration\2018_07_12_medi_phys_ball_radius_8pixel_300noise_10_8.bin'
%    fileID = fopen(savefile,'w');
%    fwrite(fileID,A,'double');
%    fclose(fileID);
%    
% %    fileID = fopen(savefile,'r');
% %    B = fread(fileID,720,'double');
% %    fclose(fileID);
%    
% %    figure
% %    plot(A(1:2:720), A(2:2:720),'r+');
% %    hold on
% %    plot(B(1:2:720), B(2:2:720),'g-');
% %    hold on
% %    plot(B(1:2:720), B(2:2:720),'r-');
% %    plot(ProjectPt2(1,:), ProjectPt2(2,:),'r-');
% return
clc;
clear;
file =  'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\圆柱球img\圆柱球结果\';%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
filefolder = dir([file,'*.img']);
imagehei = 1000;%%%
imagewid = 1000;%%%
imagesize = imagehei*imagewid;
imagenum = length(filefolder);
% for i=50:50
radius = 1000/2;%%%
volumedata = zeros(imagehei,imagewid,imagenum);

%  filename = sprintf('%.3d.proj',43);
%  a = [file,'projectdata\',filename];
%     fileID = fopen(a,'r');
% %     figure
% %     imshow(Pro_Image,[])
%     image = fread(fileID,512*512,'double');
%     tt = reshape(image,512,512);
%      figure
%      imshow(tt,[]);
%     fclose(fileID);
% return
for i=1:imagenum
  filename = filefolder(i).name;
  fileID = fopen([file,filename],'r');
  A = fread(fileID,imagesize,'double');
  fclose(fileID);
  image = reshape(A,[imagehei,imagewid]);
  volumedata(:,:,i)=image;
end;
III=volumedata(:,:,200);
% imshow(volumedata(:,:,155),[]);
eta          =  3.0/180*pi;        %平面内倾角
theta        =  3.0/180*pi;%1.2;   %左右倾角
phi          =  -3.0/180*pi;%1.5;  %前后倾角
SourcePos    = zeros(1,3);         %original X-ray source position(x,y,z)
SourcePos(1) = 8000;          
SourcePos(2) = 0;
SourcePos(3) = 0;

RotatePsiMatrix   = zeros(3,3);
RotateThetaMatrix = zeros(3,3);
RotatePhiMatrix   = zeros(3,3);

costheta = cos(theta);
sintheta = sin(theta);
RotateThetaMatrix(1,1) = costheta;
RotateThetaMatrix(1,2) = -sintheta;
RotateThetaMatrix(2,1) = sintheta;
RotateThetaMatrix(2,2) = costheta;
RotateThetaMatrix(3,3) = 1;

cosphi = cos(phi);
sinphi = sin(phi);
RotatePhiMatrix(1,1) = cosphi;
RotatePhiMatrix(1,3) = -sinphi;
RotatePhiMatrix(2,2) = 1;
RotatePhiMatrix(3,1) = sinphi;
RotatePhiMatrix(3,3) = cosphi;

RotateEtaMatrix = zeros(2,2);
cosEta = vpa(cos(eta));
sinEta = vpa(sin(eta));
RotateEtaMatrix(1,1) = cosEta;
RotateEtaMatrix(1,2) = sinEta;
RotateEtaMatrix(2,1) = -sinEta; 
RotateEtaMatrix(2,2) = cosEta;

ReverseRotateEtaMatrix = inv(RotateEtaMatrix);

OmigaAlphaBeta2PsiXYZMatrix = RotateThetaMatrix*RotatePhiMatrix;  %psi Alpha Beta 坐标系到 XYZ坐标系的转换矩阵


Proj_Hei = 1024;
Proj_Wid = 1024;
MidX = 512;
MidY = 512;
projnum = 360;
P_2D_Pt = zeros(1,2); % projection points
P_3D_Pt = zeros(1,3);
D_Vec_Matrix = zeros(Proj_Hei,Proj_Wid,3);
Length_IniValue_Matrix=zeros(Proj_Hei,Proj_Wid,4);

RotatePhiMatrix   = zeros(3,3);
RotatePsiMatrix(1,1) = 1;
RotatePsiMatrix(1,2) = 0;
RotatePsiMatrix(2,1) = 0;
RotatePsiMatrix(2,2) = 1;
RotatePsiMatrix(3,3) = 1;
SourcePos_Psi = RotatePsiMatrix*SourcePos';
for i=1:Proj_Hei       %%%%真实探测器上的y坐标
        y = i-MidY;
        P_2D_Pt(2) = y;
        for j=1:Proj_Wid   %%%%真实探测器上的x坐标
            x = j-MidX;    
            P_2D_Pt(1) = x;
            newpt = RotateEtaMatrix*P_2D_Pt';
            P_3D_Pt = [0,newpt'];
            P_3D_Pt_Psi = RotatePsiMatrix*OmigaAlphaBeta2PsiXYZMatrix*P_3D_Pt';
            DirInOmigaAlphaBeta = P_3D_Pt_Psi - SourcePos_Psi;
            DirNorm  = norm(DirInOmigaAlphaBeta);
            NormVec  = DirInOmigaAlphaBeta/DirNorm;
            a = NormVec(1)*NormVec(1) + NormVec(2)*NormVec(2);
            b = 2*(NormVec(1)*SourcePos_Psi(1)+NormVec(2)*SourcePos_Psi(2));
            c = SourcePos_Psi(1)*SourcePos_Psi(1) + SourcePos_Psi(2)*SourcePos_Psi(2) - radius*radius;
            delta = b*b - 4*a*c;
            D_Vec_Matrix(i,j,:) = NormVec;
            nlength = 0;
            if delta>0
                nlength = int16(sqrt(delta)/a);
                t1 = (-b-sqrt(delta))/(2*a);
                sam_firstpos = SourcePos_Psi + t1*NormVec;
                Length_IniValue_Matrix(i,j,2:4) = sam_firstpos;   %%%初始位置
            end;
            Length_IniValue_Matrix(i,j,1) = nlength;              %%%穿过圆柱的长度
        end;
end;
TempInitValue = zeros(1,3);

for index=162:360
    Psi = 2*pi*(index-1)/projnum;        %物体旋转的角度
    cospsi = cos(Psi);
    sinpsi = sin(Psi);
    RotatePsiMatrix(1,1) = cospsi;
    RotatePsiMatrix(1,2) = -sinpsi;
    RotatePsiMatrix(2,1) = sinpsi;
    RotatePsiMatrix(2,2) = cospsi;
    RotatePsiMatrix(3,3) = 1;

    SourcePos_Psi = RotatePsiMatrix*SourcePos';
    Pro_Image = zeros(Proj_Hei,Proj_Wid);
    
    for i=1:Proj_Hei       %%%%真实探测器上的y坐标
        %i
        for j=1:Proj_Wid   %%%%真实探测器上的x坐标
            sum = 0.00;
            nLength = Length_IniValue_Matrix(i,j,1);
            if(nLength<=0)
                continue;
            else
                TempInitValue = RotatePsiMatrix*squeeze(squeeze(Length_IniValue_Matrix(i,j,2:4)));
                NormVec = RotatePsiMatrix*squeeze(D_Vec_Matrix(i,j,:));
                for k=1:nLength
                       sam_pos = TempInitValue + (k-1)*NormVec;
                       slicepos = abs(sam_pos(3));
                       hori_pos = abs(sam_pos(2));
                       vert_pos = abs(sam_pos(1));
                       if slicepos>=398 | hori_pos>=499 | vert_pos>=499
                           continue;
                       else
                             inter_pos = floor(sam_pos)+[500,500,400]';

                             rate      = sam_pos - floor(sam_pos);
                             x = inter_pos(1);
                             y = inter_pos(2);
                             z = inter_pos(3);
%                            inter_pos = sam_pos+[256,256,250]';
%                            p_x1 = interp3(volumedata,inter_pos(1),inter_pos(2),inter_pos(3));
                             p_x1 = volumedata(x,y,z);
                             p_x2 = volumedata(x+1,y,z);
                             p_y1 = volumedata(x,y+1,z);
                             p_y2 = volumedata(x+1,y+1,z);
                             
                             intensity_1 = (p_x1*(1-rate(1))+p_x2*rate(1))*(1-rate(2))+(p_y1*(1-rate(1))+p_y2*rate(1))*rate(2);
                           
                             pp_x1 = volumedata(x,y,z+1);
                             pp_x2 = volumedata(x+1,y,z+1);
                             pp_y1 = volumedata(x,y+1,z+1);
                             pp_y2 = volumedata(x+1,y+1,z+1);
                             
                             intensity_2 = (pp_x1*(1-rate(1))+pp_x2*rate(1))*(1-rate(2))+(pp_y1*(1-rate(1))+pp_y2*rate(1))*rate(2);
                             
                             intensity = intensity_1*(1-rate(3)) + intensity_2*rate(3);

                             sum = sum + intensity;                           
                       end;
                end;
                Pro_Image(i,j) = sum;
            end;
        end;
    end;
%     filename = sprintf('%.3d.proj',index);
%     fileID = fopen([file,'projectdata\',filename],'w');
% %     figure
% %     imshow(Pro_Image,[])
%     fwrite(fileID,Pro_Image,'double');
%     fclose(fileID);
    filename = sprintf('%.3d.dcm',index);
    
    I = uint16(Pro_Image*800);
    dicomwrite(I,[file,'cylinderballincline_projdata\',filename]);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end;



return;
% figure;
% imshow(volumedata(:,:,5),[]);
% figure;
% imshow(volumedata(:,:,250),[]);
% figure;
% imshow(volumedata(:,:,480),[]);



