file =  'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\球img\';
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
% eta          =  3.0/180*pi;        %?????平面内倾角
% theta        =  4.0/180*pi;%1.2;   %????左右倾角
% phi          =  -4.0/180*pi;%1.5;  %????前后倾角
eta          =  3.0/180*pi;        %?????平面内倾角
theta        =  3.0/180*pi;%1.2;   %????左右倾角
phi          =  -3.0/180*pi;%1.5;  %????前后倾角
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

ReverseRotateEtaMatrix = inv(RotateEtaMatrix); %求逆

OmigaAlphaBeta2PsiXYZMatrix = RotateThetaMatrix*RotatePhiMatrix;  %psi Alpha Beta ???? XYZ???????? 左右*前后


Proj_Hei = 1024;
Proj_Wid = 1024;
MidX = 512;
MidY = 512;
projnum = 360;
P_2D_Pt = zeros(1,2); % projection points
P_3D_Pt = zeros(1,3);


RotatePhiMatrix   = zeros(3,3);

ballbearingpos = zeros(1,3);   %ball bearing position in (x,y,z)
ballbearingpos(1)= 265;        %ball bearing is located at the position (5.4mm 5.4mm 5.4mm)
ballbearingpos(2)= 265;
ballbearingpos(3)= -350;

ProjectPt = zeros(2,projnum); % projection points

for index=1:projnum
    Psi = 2*pi*(index-1)/projnum;        %???????
    cospsi = cos(Psi);
    sinpsi = sin(Psi);
    RotatePsiMatrix(1,1) = cospsi;
    RotatePsiMatrix(1,2) = -sinpsi;
    RotatePsiMatrix(2,1) = sinpsi;
    RotatePsiMatrix(2,2) = cospsi;
    RotatePsiMatrix(3,3) = 1;

    SourcePos_Psi = RotatePsiMatrix*SourcePos';   %%%%????Psi????????XYZ???????
    BallBearingPos= ballbearingpos'               %%%%ballbearing ?XYZ???????????
    OmegaAlphaBeta2XYZ = RotatePsiMatrix*OmigaAlphaBeta2PsiXYZMatrix;   %%%Omega_Alpha_Beta ? XYZ????????
    XYZ2OmegaAlphaBeta = inv(OmegaAlphaBeta2XYZ);
    
    BallbearingPosInOmigaAlphaBeta = XYZ2OmegaAlphaBeta*BallBearingPos;
    SourceInOmigaAlphaBeta = XYZ2OmegaAlphaBeta*SourcePos_Psi;

    %%%%%%%??????2017_3_20%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    dirvecInOmigaAlphaBeta = BallbearingPosInOmigaAlphaBeta - SourceInOmigaAlphaBeta;
    k=-SourceInOmigaAlphaBeta(1)/dirvecInOmigaAlphaBeta(1);
    
    projInAlphaBeta(1) = 0;
    projInAlphaBeta(2) = k*dirvecInOmigaAlphaBeta(2) + SourceInOmigaAlphaBeta(2);   %%%% ????????
    projInAlphaBeta(3) = k*dirvecInOmigaAlphaBeta(3) + SourceInOmigaAlphaBeta(3);   %%%% ???????? the projection point of the ball bearing in the (omega-alpha-beta coordinate system)
    
      
    ProjectPt(1,index) = projInAlphaBeta(2);  %%%???
    ProjectPt(2,index) = projInAlphaBeta(3);  %%%???
    
end;

    RotateEtaMatrix = zeros(2,2);
    cosEta = vpa(cos(eta));
    sinEta = vpa(sin(eta));
    RotateEtaMatrix(1,1) = cosEta;
    RotateEtaMatrix(1,2) = sinEta;
    RotateEtaMatrix(2,1) = -sinEta; 
    RotateEtaMatrix(2,2) = cosEta;
    
    reverse_RotateEtamatrix = inv(RotateEtaMatrix);
    ProjectPt2 = zeros(2,projnum);
%   midpt = [1024,1024];
%   noise = 4*randn(2,projnum);
 
    for i=1:projnum
        temp = ProjectPt(:,i);
        temp = vpa(RotateEtaMatrix)*temp;
        ProjectPt2(:,i) = temp;
    end;
    
    radius = 20;
    for index = 10:360
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
        
        for i=-30:1:30       %%%%???????y??
                y = floor(ProjectPt2(2,index))+i;
                P_2D_Pt(2) = y;
            for j=-30:1:30   %%%%???????x??
                x = floor(ProjectPt2(1,index))+j;    
                P_2D_Pt(1) = x;
                newpt = reverse_RotateEtamatrix*P_2D_Pt';  
                P_3D_Pt = [0,newpt'];
                P_3D_Pt_Psi = RotatePsiMatrix*OmigaAlphaBeta2PsiXYZMatrix*P_3D_Pt';  %%transfer the position in the coordinate system Omiga-alpha-beta to coordinate system x-y-z
                
                
                DirInOmigaAlphaBeta = P_3D_Pt_Psi - SourcePos_Psi; 
                DirNorm  = norm(DirInOmigaAlphaBeta);
                NormVec  = DirInOmigaAlphaBeta/DirNorm;
                
                a1 = SourcePos_Psi(1)-265;
                a2 = SourcePos_Psi(2)-265;
                a3 = SourcePos_Psi(3)+350;
                
%                 a1 = SourcePos_Psi(1)+300;
%                 a2 = SourcePos_Psi(2)+300;
%                 a3 = SourcePos_Psi(3)+300;
                
                a = 1;
                b = 2*(NormVec(1)*a1+NormVec(2)*a2+NormVec(3)*a3);
                c = a1*a1 + a2*a2 +a3*a3- radius*radius;
                delta = b*b - 4*a*c;
               
                length = 0;
                if delta>0
                    length = sqrt(delta)/a;
                end;
                Pro_Image(y+512,x+512) = length;
            end;
        end;
         filename = sprintf('%.3d.dcm',index);
%         fileID = fopen([file,'projectdata\',filename],'w');
%          figure
%          imshow(Pro_Image,[]);
         I = int16(Pro_Image*100);
         dicomwrite(I,[file,'ballprojdata1\',filename]);
         
%        dicomwrite();
%        fwrite(fileID,Pro_Image,'double');
%        fclose(fileID);
    end;
    


return;




