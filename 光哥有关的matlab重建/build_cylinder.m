cylinder=zeros(1000,1000,800);
%cylinder=uint8(cylinder);
for z=1:800
    for x=1:1000
        for y=1:1000
            if(((x-500)*(x-500)+(y-500)*(y-500))<=160000)
               cylinder(x,y,z)=0.03; 
            end;
        end;
    end;
end;
I = cylinder(:,:,400);

coordinate=zeros(8,3);
coordinate(1,1)=500;
coordinate(1,2)=100;
coordinate(1,3)=50;

coordinate(2,1)=154;
coordinate(2,2)=300;
coordinate(2,3)=150;

coordinate(3,1)=900;
coordinate(3,2)=500;
coordinate(3,3)=250;

coordinate(4,1)=154;
coordinate(4,2)=700;
coordinate(4,3)=350;

coordinate(5,1)=846;
coordinate(5,2)=700;
coordinate(5,3)=450;

coordinate(6,1)=900;
coordinate(6,2)=500;
coordinate(6,3)=550;

coordinate(7,1)=846;
coordinate(7,2)=300;
coordinate(7,3)=650;

coordinate(8,1)=500;
coordinate(8,2)=100;
coordinate(8,3)=750;


diameter=40
for i=1:8
    for z=coordinate(i,3)-diameter/2:coordinate(i,3)+diameter/2
        for x=coordinate(i,1)-diameter/2:coordinate(i,1)+diameter/2
            for y=coordinate(i,2)-diameter/2:coordinate(i,2)+diameter/2
                if(((x-coordinate(i,1))*(x-coordinate(i,1))+(y-coordinate(i,2))*(y-coordinate(i,2))+(z-coordinate(i,3))*(z-coordinate(i,3)))<=400)
                    cylinder(x,y,z)=1;
                end;
            end;
        end;
    end;
end;


% for z=1:400
%     for x=1:500
%         for y=1:500
%             if(((x-250)*(x-250)+(y-250)*(y-250)+(z-200)*(z-200))<=2500)
%                cylinder(x,y,z)=1; 
%             end;
%         end;
%     end;
% end;

II = cylinder(:,:,380);

cd 'D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\圆柱球img\圆柱球结果\'%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for z=1:800
    %z
%     filename = sprintf('%.3d.dcm',z);
%     I = cylinder(:,:,z);
%     I= im2uint16(I);
%     dicomwrite(I,['圆柱球结果\',filename]);

    I = cylinder(:,:,z);
    filename = sprintf('%.3d.img',z)
    fileID = fopen(filename, 'w');
    fwrite(fileID, I, 'double');%写切片文件
    fclose(fileID);
end;












%% 注释区
% A=fix(-14.25);

% I=dicomread('D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\圆柱球img\圆柱球结果\cylinderballprojdata\090.dcm');
% II=dicomread('D:\学习\研究生\Code\师姐重建代码\老师投影重建代码\data\圆柱球img\圆柱球结果\ellipse_cylinderballprojdata\090.dcm');
% 
% III=I-II;
% III=65536-III
% imshow(III,[])
% %dicomwrite(III,[file,'差值\',filename])



% I=rand(3)
% I=im2uint16(I)
% dicomwrite(Pro_Image,[file,'ellipse_cylinderballprojdata1\',filename])

% I = uint16(Pro_Image*1000);
% dicomwrite(I,[file,'ellipse_cylinderballprojdata1\',filename]);


% for z=1:400
%     filename = sprintf('%.3d.dcm',z);
%     I = cylinder(:,:,z);
%     dicomwrite(I,['D:\学习\研究生\Code\师姐重建代码\data\圆柱\',filename]);
% end;
