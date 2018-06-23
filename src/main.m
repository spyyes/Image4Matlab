clc
clear
% ����ͼƬ��ת���ɻҶ�ͼ
I = imread('C:/users/15199/Desktop/MatLab/img.jpg');
%imshow(I)
grayImg = rgb2gray(I);
%imshow(grayImg)
% �Ҷ�ͼ��Ƶ���ʾ
F = fftshift(fft2(grayImg));
% ����˻�����
H = getH(1/2, grayImg);
[hei,wid,~] = size(I);

%H = fftshift(fft2(m, hei, wid));
% �˻�֮���ͼ��Ƶ���ʾ
FI_psf = F .* H;
size(H)
size(I)
% �˻����ͼ��
I_psf = uint8(abs(ifft2(fftshift(FI_psf))));

% ������ͼ��
I_noise = imnoise(I_psf,'gaussian',0,0.01);

% �����Ǹ�����任����
%Ir = direct_inverse(I_noise, H);	%ֱ����任
%Ir = weina(H,grayImg, I_noise,I_psf);	%ά�ɱ任
%Ir = min2(H,grayImg, I_noise,I_psf);	%��С���
%Ir = lr(I_noise,H);			%LR�ݹ�

%imshow(Ir)

%ͳ��ͼ���ԭͼ��SSIMֵ���ж� ����
[ssimval, ssimmap] = ssim(real(Ir),grayImg);
ssimval

% ֱ����任
function [img_final] = direct_inverse(I_noise, H)
    F_Final = fftshift(fft2(I_noise));
    F_Final = F_Final ./ H;
    img_final =ifft2(fftshift(F_Final));
    img_final = uint8(img_final);
end

% ά�ɱ任
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

% ��С���˷�
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

%LR�ݹ�
function [f_out] = lr(I_noise,H)
    f = I_noise;
    imshow(f);   
    %k�ǵ���������
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

%����˻�����
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