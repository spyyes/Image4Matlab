clc
clear
% 读入图片，转换成灰度图
I = imread('C:/users/15199/Desktop/MatLab/img.jpg');
%imshow(I)
grayImg = rgb2gray(I);
%imshow(grayImg)
% 灰度图的频域表示
F = fftshift(fft2(grayImg));
% 获得退化因子
H = getH(1/2, grayImg);
[hei,wid,~] = size(I);

%H = fftshift(fft2(m, hei, wid));
% 退化之后的图像频域表示
FI_psf = F .* H;
size(H)
size(I)
% 退化后的图像
I_psf = uint8(abs(ifft2(fftshift(FI_psf))));

% 加噪后的图像
I_noise = imnoise(I_psf,'gaussian',0,0.01);

% 下面是各个逆变换函数
%Ir = direct_inverse(I_noise, H);	%直接逆变换
%Ir = weina(H,grayImg, I_noise,I_psf);	%维纳变换
%Ir = min2(H,grayImg, I_noise,I_psf);	%最小误差
%Ir = lr(I_noise,H);			%LR递归

%imshow(Ir)

%统计图像和原图的SSIM值，判断 性能
[ssimval, ssimmap] = ssim(real(Ir),grayImg);
ssimval

% 直接逆变换
function [img_final] = direct_inverse(I_noise, H)
    F_Final = fftshift(fft2(I_noise));
    F_Final = F_Final ./ H;
    img_final =ifft2(fftshift(F_Final));
    img_final = uint8(img_final);
end

% 维纳变换
function [img_final] = weina(H, I_init, I_noise, I_psf)
    N = fftshift(fft2(I_noise)) -  fftshift(fft2(I_psf));
    F_G =  fftshift(fft2(I_noise));
    F_I = fftshift(fft2(I_init));
    F_H2 = conj(H) .* H;
    F_N2 = conj(N) .* N;
    F_I2 = conj(F_I) .* F_I;
    b = F_H2 ./(F_H2 + (F_N2./F_I2));
    a = b ./ H;
    F_final = a .* F_G;
    img_final = uint8(ifft2(F_final));
end

% 最小二乘法
function [img_final] = min2(H, I_init, I_noise, I_psf)
    F_I = fftshift(fft2(I_init));
    F_G =  fftshift(fft2(I_noise));
    F_Hs = conj(H);
    gama =  0.001;
    [hei,wid,~] = size(I_init);
    p = [0 -1 0;-1 4 -1;0 -1 0];
    P = psf2otf(p,[hei,wid]);

    F_final = F_Hs ./(H.^2 + gama *(P .^2));
    F_final = F_final .* F_G;
    img_final = uint8(fftshift(ifft2(F_final)));
    %subplot(121)
    %imshow(I_noise);
    %subplot(122)
    %imshow(img_final);
end

%LR递归
function [f_out] = lr(I_noise,H)
    f = I_noise;
    imshow(f);   
    %k是迭代的轮数
    k = 1
    while k < 11
        temp1 = (ifft2(fftshift(fftshift(fft2(f)) .* H)));
        temp2 = double(I_noise) - temp1;
        temp3 = (ifft2(fftshift(fftshift(fft2(temp2)) .* H)));

        f_new = double(f) + (temp3);
        f_new(f_new<0) = 0;

        diff =  abs(sum((abs(f_new(:)))) - sum(abs(double(f(:)))))./(sum(abs(double(f(:))))+eps) ;
        f_out = uint8(f_new);
        imshow(f_out);
        f = f_new;
        k = k + 1;
    end
end

%获得退化函数
function [H] = getH(level, image)
    [m,n]=size(image);
    k=0.0025;
    H=[];
    for u=1:m
        for v=1:n
            q=((u-m/2)^2+(v-n/2)^2)^(level);
            H(u,v)=exp((-k)*q);
        end
    end
end