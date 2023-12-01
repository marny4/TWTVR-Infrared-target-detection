function [tenB, tenT,tenN] = run_Func(tenD,weight, opts)
N0 = rankN0(tenD, 0.1,size(tenD));
[M1,N1,L1] = size(tenD);
errList = 1000;
epsilon         = 1e-6;
max_iter = 100;

if ~exist('opts', 'var')
    opts = [];
end    
if isfield(opts, 'tol');         tol = opts.tol;              end
if isfield(opts, 'max_iter');    max_iter = opts.max_iter;    end
if isfield(opts, 'rho');         rho = opts.rho;              end
if isfield(opts, 'max_mu');      max_mu = opts.max_mu;        end

ext1 = 0;
extd = 0;
tenD = padarray(tenD,[ext1 ext1 extd],'symmetric');

lambda1 = opts.lambda1;
lambda2 = opts.lambda2;
lambda3 = opts.lambda3;
lambda4 = opts.lambda4;
mu = opts.mu;
mu_max = 1e6;
im_size = [M1,N1,L1];
m1 = [1 0;0 0 ];
m2 = [-1 0 ;0 0 ];
template_time(:,:,1) = m1;
template_time(:,:,2) = m2;

FDx = psf2otf([1 -1],im_size); % fourier transform of Dx in 3D cube
FDy = psf2otf([1;-1],im_size); % fourier transform of Dx in 3D cube
FDz = psf2otf(template_time,im_size); % fourier transform of Dx in 3D cube

FDxH = conj(FDx);%
FDyH = conj(FDy);
FDzH = conj(FDz);

weightTen = ones(im_size);
IL= 1./((abs(FDx).^2 + abs(FDy).^2) + 2);
IL0 = 1./((abs(FDz).^2) + 2);
%%  Initialization
sizeD= size(tenD);
B = ones(sizeD);      % B : low-rank background image
T = ones(sizeD);     % T : sparse target
N = zeros(sizeD);     % N : Noise

%Lagrange Multipliers
d1 = zeros(M1,N1,L1);
[d2,d3,d4,d5,d6,v1,v2,v3] = deal(d1);
%%
change=zeros(1,max_iter);
for iter = 1 : max_iter
%% Update Z1
    Z = run_prox_pstnn_pro(gather(B),N0,mu);
%% Update low-rank background tensor B and weightensor for B
     Lx =tenD-T-N+Z;
     Fx = IL.* (fftn(Lx+d1/mu) + FDxH.*fftn(v1 + d3/mu) + FDyH.*fftn(v2 + d4/mu));%?????
     B = ifftn(Fx);
     B = real(B);
%% Update Z2
    Z2 = prox_l1(T-d6/mu,weightTen*lambda1/mu);
    weightTen = 1./ Z2./weight;
%% Update sparse target tensor T and weightensor for T
    Lx0 =tenD-B+Z2-N;
    Fx0 = IL0.* (fftn(Lx0+d1/mu+d6/mu) + FDzH.*fftn(v3 + d5/mu));
    T = ifftn(Fx0); 
    T = real(T);
%% updata V1,V2,V3
    v1 = MySoftTh( ifftn(FDx.*Fx) - d3./mu,lambda2/mu );
    v1 = real(v1);
    v2 = MySoftTh( ifftn(FDy.*Fx) - d4./mu,lambda2/mu );
    v2 = real(v2);
    v3 = MySoftTh( ifftn(FDz.*Fx0) - d5./mu,lambda3/mu );
    v3 = real(v3);
%% Update N
     N = (mu*(tenD -B -T) + d1)/(mu+2*lambda4); 
%% Update multipliers Y1,Y2,Y3
    d1 = d1 + mu*(tenD -B - T-N);
    d2 = d2 + mu*(Z - B);
    d3 = d3 + mu*(v1- ifftn(FDx.*Fx));
    d4 = d4 + mu*(v2- ifftn(FDy.*Fx));
    d5 = d5 + mu*(v3- ifftn(FDz.*Fx0));
    d6 = d6 + mu*(Z2 - T);
%% Update mu
    if errList>0.05
        ro = 1.5;
    else
        ro = 5.5;
    end
%     ro = 1.5;
    mu = min(ro*mu,mu_max);
%% Stop Criterion
    nD = tenD(:,:,1);
    nB = B(:,:,1);
    nT = T(:,:,1);
    nN = N(:,:,1);

    errList    = norm(nD(:)-nB(:)-nT(:)-nN(:)) / norm(nD(:));
    fprintf('ASTTV-NTLA: iterations = %d   difference=%f\n', iter, errList);
    change(iter)=(errList);
    if errList < epsilon
        break;  
    end 
end
%% Output
tenB=B;tenT=T;tenN=N;