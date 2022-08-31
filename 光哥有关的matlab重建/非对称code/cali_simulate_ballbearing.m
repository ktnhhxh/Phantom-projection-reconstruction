%% 说明
% 该程序用于生成投影数据
clear;close all;
file =  'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\八球\8\';%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% filefolder = dir([file,'*.img']);
% imagehei = 512;
% imagewid = 512;
% imagesize = imagehei*imagewid;
% imagenum = length(filefolder);
% 
% radius = 512/2;
% volumedata = zeros(imagehei,imagewid,imagenum);
% 
% %  filename = sprintf('%.3d.proj',43);
% %  a = [file,'projectdata\',filename];
% %      fileID = fopen(a,'r');
% % %    figure
% % %    imshow(Pro_Image,[])
% %      image = fread(fileID,512*512,'double');
% %      tt = reshape(image,512,512);
% %      figure
% %      imshow(tt,[]);
% %      fclose(fileID);
% % return
% for i=1:imagenum
%   filename = filefolder(i).name;
%   fileID = fopen([file,filename],'r');
%   A = fread(fileID,imagesize,'double');
%   fclose(fileID);
%   image = reshape(A,[imagehei,imagewid]);
%   volumedata(:,:,i)=image;
% end;
% figure;
% imshow(volumedata(:,:,155),[]);
eta          =  1/180*pi;        %X
phi          =  2/180*pi;        %Y
theta        =  3/180*pi;       %Z
SourcePos    = zeros(1,3);      
SourcePos(1) = 8000;          
SourcePos(2) = 0;
SourcePos(3) = 0;

MidX = 510;
MidY = 516;

RotatePsiMatrix   = zeros(3,3);
RotateThetaMatrix = zeros(3,3);
RotatePhiMatrix   = zeros(3,3);

costheta = vpa(cos(theta));
sintheta = vpa(sin(theta));
RotateThetaMatrix(1,1) = costheta;
RotateThetaMatrix(1,2) = -sintheta;
RotateThetaMatrix(2,1) = sintheta;
RotateThetaMatrix(2,2) = costheta;
RotateThetaMatrix(3,3) = 1;

cosphi = vpa(cos(phi));
sinphi = vpa(sin(phi));
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

OmigaAlphaBeta2PsiXYZMatrix = RotateThetaMatrix*RotatePhiMatrix;  %psi Alpha Beta ???? XYZ????????


Proj_Hei = 1024;
Proj_Wid = 1024;

projnum = 360;
P_2D_Pt = zeros(1,2); % projection points
P_3D_Pt = zeros(1,3);


RotatePhiMatrix   = zeros(3,3);

%XYZ坐标系下的小球坐标
nBallBearingNum = 8;
BallbearingArray = zeros(nBallBearingNum,4);
BallbearingArray(:,4) = 1;
% ballbearingpos = zeros(nBallBearingNum,3);
% rad = 400-20-5;
% for i=1:nBallBearingNum
%     ballbearingpos(i,1) = fix(rad*cos(i*2*pi/nBallBearingNum));     
%     ballbearingpos(i,2) = fix(rad*sin(i*2*pi/nBallBearingNum));     
%     ballbearingpos(i,3) = 50 + (i-1)*100-400;  
% end
ballbearingpos = [350 0 400;
                  260,280,310;
                  0,360,200;
                  -280,250,80;
                  -370,0,-50;
                  -270,-260,-150;
                  0,-380,-260;
                  250,-260,-380];
BallbearingArray(:,1:3)=ballbearingpos;
              
ModelRxAngle = 4.5/180*pi;%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ModelRyAngle = 4.5/180*pi;
ModelRzAngle = 4.5/180*pi;
ModelTransX = 45;
ModelTransY = 45;
ModelTransZ = 45;
ModelRotateTransMatrix = FuncCalMatrix(ModelRxAngle,ModelRyAngle,ModelRzAngle,ModelTransX,ModelTransY,ModelTransZ);
BallbearingArrayNew = ModelRotateTransMatrix*BallbearingArray';
BallbearingArray = BallbearingArrayNew';
ballbearingpos = BallbearingArray(:,1:3);



ProjectPt = zeros(2,projnum); % projection points
ProjectPt_vec = zeros(2*nBallBearingNum,projnum);

for ball_sign = 1:8
for index=1:projnum
    Psi = 2*pi*(index-1)/projnum;        %???????
    cospsi = vpa(cos(Psi));
    sinpsi = vpa(sin(Psi));
    RotatePsiMatrix(1,1) = cospsi;
    RotatePsiMatrix(1,2) = -sinpsi;
    RotatePsiMatrix(2,1) = sinpsi;
    RotatePsiMatrix(2,2) = cospsi;
    RotatePsiMatrix(3,3) = 1;

    SourcePos_Psi = RotatePsiMatrix*SourcePos';   %%%%????Psi????????XYZ???????
    BallBearingPos = ballbearingpos(ball_sign,:)';               %%%%ballbearing ?XYZ???????????
    OmegaAlphaBeta2XYZ = RotatePsiMatrix*OmigaAlphaBeta2PsiXYZMatrix;   %%%Omega_Alpha_Beta ? XYZ????????
    
    BallbearingPosInOmigaAlphaBeta = OmegaAlphaBeta2XYZ\BallBearingPos;
    SourceInOmigaAlphaBeta = OmegaAlphaBeta2XYZ\SourcePos_Psi;
    
%     XYZ2OmegaAlphaBeta = inv(OmegaAlphaBeta2XYZ);
%     
%     BallbearingPosInOmigaAlphaBeta = XYZ2OmegaAlphaBeta*BallBearingPos;
%     SourceInOmigaAlphaBeta = XYZ2OmegaAlphaBeta*SourcePos_Psi;

    %%%%%%%??????2017_3_20%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dirvecInOmigaAlphaBeta = BallbearingPosInOmigaAlphaBeta - SourceInOmigaAlphaBeta;
    k=-SourceInOmigaAlphaBeta(1)/dirvecInOmigaAlphaBeta(1);
    
    projInAlphaBeta(1) = 0;
    projInAlphaBeta(2) = k*dirvecInOmigaAlphaBeta(2) + SourceInOmigaAlphaBeta(2);   %%%% ????????
    projInAlphaBeta(3) = k*dirvecInOmigaAlphaBeta(3) + SourceInOmigaAlphaBeta(3);   %%%% ???????? the projection point of the ball bearing in the (omega-alpha-beta coordinate system)
    
      
    ProjectPt(1,index) = projInAlphaBeta(2);  %%%???
    ProjectPt(2,index) = projInAlphaBeta(3);  %%%???
    
    
end

ProjectPt_vec(ball_sign*2-1:ball_sign*2,:) = ProjectPt;

end

%     RotateEtaMatrix = zeros(2,2);
%     cosEta = vpa(cos(eta));
%     sinEta = vpa(sin(eta));
%     RotateEtaMatrix(1,1) = cosEta;
%     RotateEtaMatrix(1,2) = sinEta;
%     RotateEtaMatrix(2,1) = -sinEta; 
%     RotateEtaMatrix(2,2) = cosEta;
    
    reverse_RotateEtamatrix = inv(RotateEtaMatrix);
    ProjectPt_vec2 = zeros(2*nBallBearingNum,projnum);
    %ProjectPt2 = zeros(2,projnum);
%   midpt = [1024,1024];
%   noise = 4*randn(2,projnum);
 for ball_sign = 1:nBallBearingNum
    for i=1:projnum
        temp = ProjectPt_vec(ball_sign*2-1:ball_sign*2,i);
        temp = vpa(RotateEtaMatrix)*temp;
        ProjectPt_vec2(ball_sign*2-1:ball_sign*2,i) = temp;
    end
 end
    
 %AAA = ProjectPt_vec2+512;
 AAA = ProjectPt_vec2;
 AAA(1:2:15,:) = AAA(1:2:15,:)+MidX;
 AAA(2:2:16,:) = AAA(2:2:16,:)+MidY;

cd  'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\八球\8\'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ProjectPt_vec2save=reshape(ProjectPt_vec2,[],1);
save('ProjectPt_vec2.mat', 'ProjectPt_vec2save')
 
%  AA = reshape(AAA,[],1);
%  filename='E:\校正程序2\Bin\Config\pt_irregular.bin';
%  fileID = fopen(filename,'w');
%  fwrite(fileID,AA,'double');
%  fclose(fileID);
 
 
    radius = 20;
  
for index = 1:360
    Pro_Image = zeros(Proj_Hei,Proj_Wid);
    Psi = 2*pi*(index-1)/projnum;        %???????
    cospsi = cos(Psi);
    sinpsi = sin(Psi);
    RotatePsiMatrix(1,1) = cospsi;
    RotatePsiMatrix(1,2) = -sinpsi;
    RotatePsiMatrix(2,1) = sinpsi;
    RotatePsiMatrix(2,2) = cospsi;
    RotatePsiMatrix(3,3) = 1;
    SourcePos_Psi = RotatePsiMatrix*SourcePos';
    
    Bound = radius+10;
    for ball_sign = 1:8
        for i=-Bound:1:Bound       %%%%???????y??
                y = floor(ProjectPt_vec2(ball_sign*2,index))+i;
                P_2D_Pt(2) = y;
            for j=-Bound:1:Bound   %%%%???????x??
                x = floor(ProjectPt_vec2(ball_sign*2-1,index))+j;    
                P_2D_Pt(1) = x;
                %newpt = reverse_RotateEtamatrix*P_2D_Pt'; 
                
                newpt = RotateEtaMatrix\P_2D_Pt'; 
                P_3D_Pt = [0,newpt'];
                P_3D_Pt_Psi = RotatePsiMatrix*OmigaAlphaBeta2PsiXYZMatrix*P_3D_Pt';  %%transfer the position in the coordinate system Omiga-alpha-beta to coordinate system x-y-z
                
               
                
                DirInOmigaAlphaBeta = P_3D_Pt_Psi - SourcePos_Psi; 
                DirNorm  = norm(DirInOmigaAlphaBeta);
                NormVec  = DirInOmigaAlphaBeta/DirNorm;
                
                a1 = SourcePos_Psi(1)-ballbearingpos(ball_sign,1);
                a2 = SourcePos_Psi(2)-ballbearingpos(ball_sign,2);
                a3 = SourcePos_Psi(3)-ballbearingpos(ball_sign,3);
                
                a = 1;
                b = 2*(NormVec(1)*a1+NormVec(2)*a2+NormVec(3)*a3);
                c = a1*a1 + a2*a2 +a3*a3- radius*radius;
                delta = b*b - 4*a*c;
               
                length = 0;
                if delta>0
                    length = sqrt(delta)/a;
                end
               
               
%                 if i==0 && j==0
%                     P_2D_Pt
%                     x+MidX
%                     y+MidY
%                     Proj_Hei-MidY-y
%                 end
%                 
                
                Pro_Image(y+MidY,x+MidX) = length;
                %Pro_Image(x+MidX,y+MidY) = length;
                %Pro_Image(x+MidX,Proj_Hei-MidY-y) = length;
                
            end
        end
%          I = int16(32767-Pro_Image*1000);
%          figure
%          imshow(I,[])
    end
        filename = sprintf('%.3d.dcm',index);
%         fileID = fopen([file,'projectdata\',filename],'w');
%          figure
%          imshow(Pro_Image,[]);
         I = int16(32767-Pro_Image*1000);
%          figure
%          imshow(I,[])
         dicomwrite(I,[file,'dcmimage\',filename]);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
%        dicomwrite();
%        fwrite(fileID,Pro_Image,'double');
%        fclose(fileID);
end


