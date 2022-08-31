height = 500;
width  = 500;
slice  = 400;
%volume用来读取老师数据
volume = zeros(height,width,slice);
dim = size(volume);
cd 'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\圆柱球img'
for i=1:dim(3)
  filename = sprintf('%.3d.img',i);
  fileID = fopen(filename,'r');
  temp=fread(fileID,'double');
  temp=reshape(temp,height,width);
  volume(:,:,i)=temp;
  %I=volume(:,:,i);
%   figure
%   imshow(volume(:,:,i),[])
  fclose(fileID);
end;
%volume = zeros(516,516);
III=volume(:,:,200)
%imshow(proj(:,:),[])

A=cos(pi/6)