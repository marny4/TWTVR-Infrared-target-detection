function [lambda_1, lambda_2] = structure_tensor_lambda(img, sz)% �����˹������̫��Ҳ���Ǹ�ģ��������ɸ��౳���龯

G = fspecial('gaussian', [sz sz], 2); % Gaussian kernel ��sz��Hsize=3��sigma=2��sigmaԽСԽ����,������ǰȥ������?
u = imfilter(img, G, 'symmetric');

[Gx, Gy] = gradient(u); % �����ߴ�Ϊ2�ĸ�˹�˲���õĺ����ݶ�

K = fspecial('gaussian', [sz sz],9); % Gaussian kernel��sz��Hsize=3��sigma=9��
J_11 = imfilter(Gx.^2, K, 'symmetric'); %������ÿ��Ԫ�ض�ƽ��
J_12 = imfilter(Gx.*Gy, K, 'symmetric');
J_21 = J_12;
J_22 = imfilter(Gy.^2, K, 'symmetric');   

sqrt_delta = sqrt((J_11 - J_22).^2 + 4*J_12.^2);
lambda_1 = 0.5*(J_11 + J_22 + sqrt_delta);
lambda_2 = 0.5*(J_11 + J_22 - sqrt_delta);


