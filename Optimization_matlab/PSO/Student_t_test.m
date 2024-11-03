clc;clear;close all;
PSO = [8.3885 8.3885 8.3891 8.389 8.3862 8.389 8.3888 8.389 8.389 8.3891];
GA = [8.3878 8.3883 8.3586 8.3767 8.3878 8.3874 8.3835 8.3685 8.383 8.389];
A = PSO; B = GA;
mean_A = mean(A);     mean_B = mean(B);
var_A = var(A);       var_B = var(B);
nA = length(A);       nB = length(B);
alpha = 0.05;
%% Test whether variances are equal
% H0 : σA = σB
% H1 : σA /= σB
F = var_A/var_B;
F_upper = 4.03;           % F(1-α/2;nA-1;nB-1) =  F(0.025;9;9) = 4.03 查表
F_lower = 1/F_upper;      % F(α/2;nA-1;nB-1) = F(0.975;9;9) = 1/F(0.025;9;9)
if F <= F_upper && F >= F_lower
    equal = 'Y';          % variances are equal
else
    equal = 'N';          % variances are not equal
end
%% Student t-test
% H0 : μA - μB <= 0
% H1 : μA - μB > 0
if equal == 'Y'
    sD =  sqrt(((nA-1)*var_A + (nB-1)*var_B)/(nA + nB -2))*sqrt((1/nA) + (1/nB));
    t = (mean_A - mean_B)/sD;
    df = nA + nB -2;
    t_one_tail = 1.734;        % t(1-α,nA+nB-2) = t(0.95,18) = 1.734 查表
elseif equal == 'N'
    sD = sqrt((var_A/nA) + (var_B/nB));
    t = (mean_A - mean_B)/sD;
    df = (((var_A/nA) + (var_B/nB))^2)/((((var_A/nA)^2)/(nA-1)) + (((var_B/nB)^2)/(nB-1)));
    t_one_tail = 1.734;        % t(1-α,nA+nB-2) = t(0.95,18) = 1.734 查表
end
%% Test result
if (t - t_one_tail) > 1
    disp("t = " + num2str(t) + " > t (one tail) = " + num2str(t_one_tail))
    fprintf(2,'=> PSO are significantly greater than GA\n')
elseif (t - t_one_tail) > 0 && (t - t_one_tail) < 1
    disp("t = " + num2str(t) + " > t (one tail) = " + num2str(t_one_tail))
    fprintf(2,'=> The difference between PSO and GA is not significantly\n')
elseif (t - t_one_tail) < -1
    disp("t = " + num2str(t) + " < t (one tail) = " + num2str(t_one_tail))
    fprintf(2,'=> GA are significantly greater than PSO\n')
elseif (t - t_one_tail) <0 && (t - t_one_tail) > -1
    disp("t = " + num2str(t) + " < t (one tail) = " + num2str(t_one_tail))
    fprintf(2,'=> The difference between PSO and GA is not significantly\n')
end