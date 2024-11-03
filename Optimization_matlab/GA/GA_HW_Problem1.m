clc;clear;close all
%% step1 Initialization
% Precision = 10^-4
% -1 <= x <= 1 , 將x切成30000等分 , 2^14 < 30000 < 2^15 (x採用15-bit chromosomes)
% -2 <= y <= 1 , 將y切成40000等分 , 2^15 < 40000 < 2^16 (y採用16-bit chromosomes)
% 31-bit chromosome:15 bits for variable x and 16 bits for variable y
x_upper =  1; x_lower = -1;            % x值上下限
y_upper =  1; y_lower = -2;            % y值上下限
xy_upper = 1;                          % x+y值上限
genes_x = 15; genes_y = 16;
genes = genes_x + genes_y;
popsize = 800;
Chrom_2 = zeros(popsize,genes);        % Chromosome Binary  2進制
Chrom_10 = zeros(popsize,genes);       % Chromosome Decimal 10進制(未加總的)
CR = 0.9;MR = 0.2;                     % Crossover Rate & Mutation Rate
Iteration = 50;                        %Iteration(迭代次數)
convergence_threshold = 1e-4;          % Termination condition : Convergence
convergence_max_times = 7;             % 如果Fitness值前後相差不到0.0001的次數達到7次，則達到收斂條件，停止迭代
%% step2 Evaluation
tic;
i = 1;
while i <= popsize
    for j = 1:genes_x
        k = genes_x-j;
        Chrom_2(i,j) = round(rand);
        Chrom_10(i,j) = Chrom_2(i,j)*power(2,k);
    end

    for j = genes_y:genes
        k = genes-j;
        Chrom_2(i,j) = round(rand);
        Chrom_10(i,j) = Chrom_2(i,j)*power(2,k);
    end

    Chrom_10_sum_x(i) = sum(Chrom_10(i,1:genes_x));      % Chromosome 10進制 x值
    Chrom_10_sum_y(i) = sum(Chrom_10(i,genes_y:genes));  % Chromosome 10進制 y值
    x(i) = x_lower + (Chrom_10_sum_x(i)*(x_upper-x_lower))/(power(2,genes_x)-1);
    y(i) = y_lower + (Chrom_10_sum_y(i)*(y_upper-y_lower))/(power(2,genes_y)-1);
    f(i) = exp(-0.1*(x(i)^2 + y(i)^2)) + exp(cos(4*pi*x(i)) + cos(2*pi*y(i)));

    % Constraint Handling ==> Pre-censoring
    if (x(i) < x_lower) || (x(i) > x_upper)
        Chrom_2(i,:) = []; Chrom_10(i,:) = [];
        Chrom_10_sum_x(i) = []; Chrom_10_sum_y(i) = [];
        x(i) = []; y(i) = []; f(i) = []; i = i;

    elseif (y(i) < y_lower) || (y(i) > y_upper)
        Chrom_2(i,:) = []; Chrom_10(i,:) = [];
        Chrom_10_sum_x(i) = []; Chrom_10_sum_y(i) = [];
        x(i) = []; y(i) = []; f(i) = []; i = i;

    elseif (x(i) + y(i)) > xy_upper
        Chrom_2(i,:) = []; Chrom_10(i,:) = [];
        Chrom_10_sum_x(i) = []; Chrom_10_sum_y(i) = [];
        x(i) = []; y(i) = []; f(i) = []; i = i;

    else
        i = i+1;
    end
end
clear i j k
%% step3 Selection (Roulette wheel)
Generation = Chrom_2;                    %將Chrom_2當作第一代父母(2進制)
Generation_10 = zeros(popsize,genes);    %第一代父母(10進制)
child = zeros(popsize,genes);            %初始化_小孩
Fitness = zeros(1,Iteration+1);          %各世代表現最佳的f(x,y)
Gen_best_child_x = zeros(1,Iteration+1); %各世代表現最佳的f(x,y)對應的x
Gen_best_child_y = zeros(1,Iteration+1); %各世代表現最佳的f(x,y)對應的y
convergence_times = 0;                   %初始收斂次數
% Start Iteration
for It = 1:Iteration
    for I = 1:popsize
        for j = 1:genes_x
            k = genes_x-j;
            Generation_10(I,j) = Generation(I,j)*power(2,k);
        end

        for j = genes_y:genes
            k = genes-j;
            Generation_10(I,j) = Generation(I,j)*power(2,k);
        end
        Gen_sum_x(I) = sum(Generation_10(I,1:genes_x));
        Gen_sum_y(I) = sum(Generation_10(I,genes_y:genes));
        x_sele(I) = x_lower + (Gen_sum_x(I)*(x_upper-x_lower))/(power(2,genes_x)-1);
        y_sele(I) = y_lower + (Gen_sum_y(I)*(y_upper-y_lower))/(power(2,genes_y)-1);
        f_sele(I) = exp(-0.1*(x_sele(I)^2 + y_sele(I)^2)) + exp(cos(4*pi*x_sele(I)) + cos(2*pi*y_sele(I)));
    end
    z = 1;
    while z < popsize
        disp("Time: "+ datestr(now,'HH:MM:SS.FFF') + "; Iteration: " + num2str(It) + "; Generate Child " + num2str(z))
        f_pick = f_sele';                         %f_pick : 用作挑選父母的f(x,y)
        for i = 1:2                               %for i = 1:2 共計要挑選一對雙親
            f_pick = f_pick;
            f_pick = f_pick - min(f_pick);
            Nor_f = f_pick/sum(f_pick);           % Normalization
            Cum = zeros(1,length(Nor_f));         % Cumulation
            for k = 1:length(f_pick)
                Cum(k) = sum(Nor_f(1:k,1));
            end
            randn(i) = round(rand,4);             %隨機生成一個0~1的數字來決定父母
            for j = 1:length(Cum)-1
                if randn(i) < Cum(1)
                    pick(i) = 1;                  %pick:找出要挑選父母的所在位置
                elseif randn(i) >= Cum(j) && randn(i) < Cum(j+1)
                    pick(i) = j+1;
                end
            end
            parents{i} = Generation(pick(i),:);   %找到父母位置後,從Generation裡挑選出父母
            f_pick(pick(i)) = [];                 %挑選到第一位家長後,將他從f_pick裡刪除,避免下次挑到重覆的家長
        end
        clear i j k randn
        %% step4 Crossover (One-point crossover)
        randn_cross = round(rand,4);              %隨機生成一個0~1的數字來決定是否生小孩
        if randn_cross < CR
            Cross_location = ceil((round(rand,4))*(genes-1));  %隨機生成一個0~1數字決定One-point crossover的位置
            child_1 = [parents{1}(1:Cross_location) parents{2}(Cross_location+1:end)];
            child_2 = [parents{2}(1:Cross_location) parents{1}(Cross_location+1:end)];
            %% step5 Mutation (One-bit-mutation)
            rand_mutation_1 = round(rand,4);      %隨機生成一個0~1的數字來決定第一個孩子是否變異
            rand_mutation_2 = round(rand,4);      %隨機生成一個0~1的數字來決定第二個孩子是否變異
            if rand_mutation_1 < MR
                Mutation_location_1 = ceil((round(rand,4))*(genes)); %隨機生成一個0~1的數字決定變異位置

                if Mutation_location_1 == 0
                    Mutation_location_1 = 1;
                end

                if child_1(Mutation_location_1) == 0
                    child_1(Mutation_location_1) = 1;
                else
                    child_1(Mutation_location_1) = 0;
                end
            else
                child_1 = child_1;
            end

            if rand_mutation_2 < MR
                Mutation_location_2 = ceil((round(rand,4))*(genes)); %隨機生成一個0~1的數字決定變異位置

                if Mutation_location_2 == 0
                    Mutation_location_2 = 1;
                end

                if child_2(Mutation_location_2) == 0
                    child_2(Mutation_location_2) = 1;
                else
                    child_2(Mutation_location_2) = 0;
                end

            else
                child_2 = child_2;
            end
            % Constraint Handling ==> Pre-censoring (生下來的小孩(變異)也可能超出範圍)
            for i = 1:genes_x
                k = genes_x-i;
                child_1_x(i) = child_1(i)*power(2,k);
                child_2_x(i) = child_2(i)*power(2,k);
            end

            for i = genes_y:genes
                k = genes-i;
                child_1_y(i) = child_2(i)*power(2,k);
                child_2_y(i) = child_2(i)*power(2,k);
            end
            child_1_x = sum(child_1_x(1:genes_x));
            child_1_y = sum(child_1_y(genes_y:genes));
            child_2_x = sum(child_2_x(1:genes_x));
            child_2_y = sum(child_2_y(genes_y:genes));

            child_1_x = x_lower + (child_1_x*(x_upper-x_lower))/(power(2,genes_x)-1);
            child_1_y = y_lower + (child_1_y*(y_upper-y_lower))/(power(2,genes_y)-1);
            child_2_x = x_lower + (child_2_x*(x_upper-x_lower))/(power(2,genes_x)-1);
            child_2_y = y_lower + (child_2_y*(y_upper-y_lower))/(power(2,genes_y)-1);
            clear i k

            if (child_1_x < x_lower) || (child_1_x > x_upper || child_2_x < x_lower || child_2_x > x_upper)
                z = z;

            elseif (child_1_y < y_lower) || (child_1_y > y_upper || child_2_y < y_lower || child_2_y > y_upper)
                z = z;

            elseif ((child_1_x + child_1_y) > xy_upper) || ((child_2_x + child_2_y) > xy_upper)
                z = z;

            else
                child(z,:) = child_1;
                child(z+1,:) = child_2;
                z = z+2;
            end

        else
            z = z;          %對應上面randn_cross , 若randn_cross < CR 則不會生出小孩,次數也不會增加
        end
    end
    Gen_child_10 = zeros(popsize,genes);
    for i = 1:popsize
        for j = 1:genes_x
            k = genes_x-j;
            Gen_child_10(i,j) = child(i,j)*power(2,k);
        end

        for j = genes_y:genes
            k = genes-j;
            Gen_child_10(i,j) = child(i,j)*power(2,k);
        end
        Gen_child_sum_x(i) = sum(Gen_child_10(i,1:genes_x));
        Gen_child_sum_y(i) = sum(Gen_child_10(i,genes_y:genes));
        x_child(i) = x_lower + (Gen_child_sum_x(i)*(x_upper-x_lower))/(power(2,genes_x)-1);
        y_child(i) = y_lower + (Gen_child_sum_y(i)*(y_upper-y_lower))/(power(2,genes_y)-1);
        f_child(i) = exp(-0.1*(x_child(i)^2 + y_child(i)^2)) + exp(cos(4*pi*x_child(i)) + cos(2*pi*y_child(i)));
    end
    clear i j k
    %將每個世代裡最佳的f(x,y)和對應的最佳x和y記錄下來
    [max_f_child maxloca] = max(f_child);
    if max_f_child >= Fitness(It)
        Fitness(It+1) = max_f_child;
        Gen_best_child_x(It+1) = x_child(maxloca);
        Gen_best_child_y(It+1) = y_child(maxloca);
    else
        Fitness(It+1) = Fitness(It);
        Gen_best_child_x(It+1) = Gen_best_child_x(It);
        Gen_best_child_y(It+1) = Gen_best_child_y(It);
    end

    % Termination condition : Convergence
    if (Fitness(It+1) - Fitness(It)) < convergence_threshold
        convergence_times = convergence_times + 1;
        if convergence_times > convergence_max_times
            stop = It+1;
            break
        end
    end

    Generation = child;
end
clear I It z
time = toc;
%% Final Optimization Result
%在每個世代裡依照f(x,y)挑出最好的那個世代和對應的x和y值作為最佳解
[maxF,Loca] = max(Fitness);
Best_Variable_x = Gen_best_child_x(Loca);
Best_Variable_y = Gen_best_child_y(Loca);
disp([newline,'Best Variable x : ',num2str(Best_Variable_x),newline,...
    'Best Variable y : ',num2str(Best_Variable_y),newline,...
    'Maximum of Objective Function : ',num2str(maxF),newline,...
    'Iteration times : ',num2str(stop-1),newline,...
    'Total time spent for GA = ',num2str(time),' sec',])
%% Evolution History Plot
if convergence_times > convergence_max_times
    convergence  = Fitness(2:stop);
    Ite = 1:length(convergence);
else
    convergence  = Fitness(2:Iteration+1);
    Ite = 1:Iteration;
end
figure;
plot(Ite,convergence,'xr-');xlabel('Iterations');ylabel('Fitness');
xlim([0 length(Ite)]);ylim([8.2 8.4]);title('Evolution History');
legend('best record so far');
set(gca,'YTick',8:0.01:8.5);
set(gca,'ygrid','on','Gridalpha',0.4);