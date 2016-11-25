# Linear-Regression-Classification
formulating the pattern recognition problem in terms of linear regression

function [ id ] = LRC( D,class_pinv_M,y,Dlabels )
%------------------------------------------------------------------------
% LRC classification function
coef         =  class_pinv_M*y;
for ci = 1:max(Dlabels)
    coef_c   =  coef(Dlabels==ci);   %取出对应i类的系数
    Dc       =  D(:,Dlabels==ci);    %取出对应i类的训练字典
    error(ci) = norm(y-Dc*coef_c,2)^2/sum(coef_c.*coef_c);
end

index      =  find(error==min(error));
id         =  index(1);
