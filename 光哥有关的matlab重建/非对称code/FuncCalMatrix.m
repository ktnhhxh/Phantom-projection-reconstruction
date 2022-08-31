%% 计算转换矩阵
% 顺序：XZY
% 返回矩阵大小：4*4
% 传入参数为弧度值

function RotateTransmatrix = FuncCalMatrix(eta,phi,theta,TransX,TransY,TransZ)
RotateEtaMatrix   = zeros(4,4);
RotateThetaMatrix = zeros(4,4);
RotatePhiMatrix   = zeros(4,4);

cosphi = cos(phi);
sinphi = sin(phi);
RotatePhiMatrix(1,1) = cosphi;
RotatePhiMatrix(1,2) = -sinphi;
RotatePhiMatrix(2,1) = sinphi;
RotatePhiMatrix(2,2) = cosphi;
RotatePhiMatrix(3,3) = 1;

costheta = cos(theta);
sintheta = sin(theta);

RotateThetaMatrix(1,1) = costheta;
RotateThetaMatrix(1,3) = -sintheta;
RotateThetaMatrix(2,2) = 1;
RotateThetaMatrix(3,1) = sintheta;
RotateThetaMatrix(3,3) = costheta;

coseta = cos(eta);
sineta = sin(eta);

RotateEtaMatrix(1,1) = 1.0;
RotateEtaMatrix(2,2) = coseta;
RotateEtaMatrix(2,3) = -sineta;
RotateEtaMatrix(3,2) = sineta;
RotateEtaMatrix(3,3) = coseta;


RotateTransmatrix = RotatePhiMatrix*RotateThetaMatrix*RotateEtaMatrix;  %XYZ坐标系到OmegaAlphaBeta坐标系的转换矩阵

TranslationVector = zeros(1,4);
TranslationVector(1,1) = TransX; % 4000
TranslationVector(1,2) = TransY;
TranslationVector(1,3) = TransZ;  
TranslationVector(1,4) = 1.0;

RotateTransmatrix(:,4) = TranslationVector;
end