function [lambda_1, lambda_2] = structure_tensor_lambda(img, sz)% 如果高斯两参数太大，也就是更模糊，会造成更多背景虚警

G = fspecial('gaussian', [sz sz], 2); % Gaussian kernel ，sz：Hsize=3。sigma=2。sigma越小越尖锐,用来提前去除噪声?
u = imfilter(img, G, 'symmetric');

[Gx, Gy] = gradient(u); % 经过尺寸为2的高斯滤波求得的横纵梯度

K = fspecial('gaussian', [sz sz],9); % Gaussian kernel，sz：Hsize=3。sigma=9。
J_11 = imfilter(Gx.^2, K, 'symmetric'); %矩阵中每个元素都平方
J_12 = imfilter(Gx.*Gy, K, 'symmetric');
J_21 = J_12;
J_22 = imfilter(Gy.^2, K, 'symmetric');   

sqrt_delta = sqrt((J_11 - J_22).^2 + 4*J_12.^2);
lambda_1 = 0.5*(J_11 + J_22 + sqrt_delta);
lambda_2 = 0.5*(J_11 + J_22 - sqrt_delta);


