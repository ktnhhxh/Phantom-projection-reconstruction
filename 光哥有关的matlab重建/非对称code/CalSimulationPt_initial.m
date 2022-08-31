%clear;
%生成解析的圆心，也就是一点误差也没有的
nBallBearingNum = 8;
nAngleNum = 360;
BallbearingArray = zeros(nBallBearingNum,4);
BallbearingArray(:,4) = 1;
BallCoordIrregular = [350 0 400;
                      260,280,310;
                      0,360,200;
                      -280,250,80;
                      -370,0,-50;
                      -270,-260,-150;
                      0,-380,-260;
                      250,-260,-380];
BallbearingArray(:,1:3) = BallCoordIrregular;         

ModelRxAngle = 0/180*pi;
ModelRyAngle = 0/180*pi;
ModelRzAngle = 6/180*pi;
ModelTransX = 200;
ModelTransY = 600;
ModelTransZ = 0;
ModelRotateTransMatrix = FuncCalMatrix(ModelRxAngle,ModelRyAngle,ModelRzAngle,ModelTransX,ModelTransY,ModelTransZ);
BallbearingArrayNew = ModelRotateTransMatrix*BallbearingArray';
BallbearingArray = BallbearingArrayNew';

XYZ2OmegaAlphaBetaMatrix = zeros(4,4);
BallbearingArray_In_OmeAlphaBeta = zeros(nBallBearingNum,4);

SourcePos= zeros(1,3);  
SourcePos(1) = 8000;
SourcePos(2) = 512;
SourcePos(3) = 512;

for phi = 0:nAngleNum-1
    
    eta   = 1;%绕X
    theta = 2;%绕Y
    %phi   = 0;%绕Z

    RotateEtaMatrix   = zeros(4,4);
    RotateThetaMatrix = zeros(4,4);
    RotatePhiMatrix   = zeros(4,4);
    cosphi = cos(phi/180*pi);
    sinphi = sin(phi/180*pi);
    RotatePhiMatrix(1,1) = cosphi;
    RotatePhiMatrix(1,2) = -sinphi;
    RotatePhiMatrix(2,1) = sinphi;
    RotatePhiMatrix(2,2) = cosphi;
    RotatePhiMatrix(3,3) = 1;
    costheta = cos(theta/180*pi);
    sintheta = sin(theta/180*pi);
    RotateThetaMatrix(1,1) = costheta;
    RotateThetaMatrix(1,3) = -sintheta;
    RotateThetaMatrix(2,2) = 1;
    RotateThetaMatrix(3,1) = sintheta;
    RotateThetaMatrix(3,3) = costheta;
    coseta = cos(eta/180*pi);
    sineta = sin(eta/180*pi);
    RotateEtaMatrix(1,1) = 1.0;
    RotateEtaMatrix(2,2) = coseta;
    RotateEtaMatrix(2,3) = -sineta;
    RotateEtaMatrix(3,2) = sineta;
    RotateEtaMatrix(3,3) = coseta;
    XYZ2OmegaAlphaBetaMatrix = RotatePhiMatrix*RotateThetaMatrix*RotateEtaMatrix;
    
    RotateMatrix = XYZ2OmegaAlphaBetaMatrix(1:3,1:3);
    RotateMatrix_inv = inv(RotateMatrix);
    XYZ2OmegaAlphaBetaMatrix(1:3,1:3) = RotateMatrix_inv;

    TranslationVector = zeros(1,4);
    TranslationVector(1,1) = 0; 
    TranslationVector(1,2) = 512;
    TranslationVector(1,3) = 512;  
    TranslationVector(1,4) = 1.0;

    XYZ2OmegaAlphaBetaMatrix(:,4) = TranslationVector;

    for i=1:nBallBearingNum
            tmp = XYZ2OmegaAlphaBetaMatrix*BallbearingArray(i,:)';
            BallbearingArray_In_OmeAlphaBeta(i,:) = tmp';
    end

    Proj_UV_Data = zeros(nBallBearingNum,2); 
    SourcePos_ext = repmat(SourcePos,nBallBearingNum,1);
    kk = SourcePos_ext(:,1)./(SourcePos_ext(:,1)-BallbearingArray_In_OmeAlphaBeta(:,1));
    Proj_UV_Data(:,1) = -(SourcePos_ext(:,2)-BallbearingArray_In_OmeAlphaBeta(:,2)).*kk + SourcePos_ext(:,2);
    Proj_UV_Data(:,2) = -(SourcePos_ext(:,3)-BallbearingArray_In_OmeAlphaBeta(:,3)).*kk + SourcePos_ext(:,3);
    
    
    k = abs(phi-360)-1;
    PtDataIndex = k*nBallBearingNum*2+1;
    %PtDataIndex = Psi*nBallBearingNum*2+1;
    tmp = reshape(Proj_UV_Data',[],1);
    PtData(PtDataIndex:PtDataIndex+nBallBearingNum*2-1) = tmp;
    
end


A = reshape(PtData,16,[],1);


filename = 'E:\校正程序2\Bin\Config\pt_irregular_1_2_0.bin';
fileID = fopen(filename,'r');
fwrite(fileID,A,'double');
fclose(fileID);
%Data2 = reshape(Data,16,[]);
