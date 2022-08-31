% cylinder=zeros(500,500,400);
% %cylinder=uint8(cylinder);
% radius=500
% for z=1:400
%     for x=1:radius
%         for y=1:radius
%             if(((x-15)*(x-15)+(y-15)*(y-15)+(z-100)*(z-100))<=100)
%                cylinder(x,y,z)=1; 
%             end;
%         end;
%     end;
% end;
% II = cylinder(:,:,200);
% 
% cd 'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\球img'
% for z=1:400
% %     filename = sprintf('%.3d.dcm',z);
% %     I = cylinder(:,:,z);
% %     dicomwrite(I,['D:\学习\研究生\Code\师姐重建代码\data\圆柱\',filename]);
% 
%     I = cylinder(:,:,z);
%     filename = sprintf('%.3d.img',z)
%     fileID = fopen(filename, 'w');
%     fwrite(fileID, I, 'double');%写切片文件
%     fclose(fileID);
% end;

I= D_Vec_Matrix(1000,1000,:)
II=D_Vec_Matrix_ellipse(1000,1000,:)