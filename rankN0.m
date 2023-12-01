function N = rankN0(X, ratioN,dim)
    
    svd0 = reshape(pagesvd(X),[dim(1),dim(3)]);
    
    [desS, ~] = sort(svd0, 'descend'); 
    ratioVec = desS / desS(1);

    N = zeros([dim(3),1]);
    for i = 1:dim(3)
        idxArr = find(ratioVec(:,i) < ratioN); 
        if idxArr(1) > 1
            N(i) = idxArr(1) - 1; 
        else
            N(i) = 1;
        end
    end