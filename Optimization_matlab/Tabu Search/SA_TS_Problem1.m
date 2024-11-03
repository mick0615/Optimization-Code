clc;clear;close all
%% Parameters
distance = xlsread('SA TS Problems.xlsx');
distance = distance(2:end,2:end);
node = size(distance,1);
Tabu_list = zeros(node,node);
tabu_tenure = 3;
aspiration_criterion = 5;
swap_move = 15;
iteration = 200;
Best_Objective = zeros(iteration,1);
Best_permutation = zeros(iteration,node);
permutation = randperm(node);
pre_tabu = ones(tabu_tenure+1,2);
Swap = zeros(swap_move,2);
Best_record = zeros(iteration + 1,1);
Best_record(1,1) = inf;
Best_record_permutation = zeros(iteration + 1,node);
convergence_threshold = 0;      
convergence_max_times = 35; 
convergence_times = 0;
%% Start Iteration
tic;
for I = 1:iteration
    pre_tabu = [pre_tabu(end, :); pre_tabu(1:end-1, :)]; %最後一項跑到最前面
    for i = 1:swap_move
        disp("Time: "+ datestr(now,'HH:MM:SS.FFF') + "; Iteration: " + num2str(I) + "; swap_move " + num2str(i))
        per = permutation;
        s = 1;
        while s <= swap_move
            value1 = randi(node);  value2 = randi(node);
            while value2 == value1
                value2 = randi(node);
            end

            state = 0;
            for j = 1:swap_move
                if (Swap(j,1) == min(value1,value2) && Swap(j,2) == max(value1,value2))
                    state = 1;
                end
            end

            if state == 0
                Swap(i,:) = sort([value1,value2]);
                s = s + 1;
            else
                s = s;
            end
        end
        index1 = find(per == Swap(i,1));  index2 = find(per == Swap(i,2));
        temp = per(index1);
        per(index1) = per(index2);
        per(index2) = temp;
        iter_per(i,:) = per;
        Obj_Value(i,:) = Obj_dis(per,distance);
        Obj_Value = Obj_Value + Tabu_list(max(value1,value2),min(value1,value2));
    end
    list = [Swap Obj_Value iter_per];
    list = sortrows(list, 3);
    k = 1;
    while (Tabu_list(list(k,1),list(k,2)) ~= 0 && (Best_Objective(I) - list(1,3)) < aspiration_criterion)
        k = k + 1;
    end
    Best_permutation(I,:) = list(k,4:size(list,2));
    Best_Objective(I) = Obj_dis(Best_permutation(I,:),distance);
    swap = list(k,1:2);
    pre_tabu(1,:) = swap;
    for t = 1:tabu_tenure + 1
        tt = (tabu_tenure + 1) - t;
        Tabu_list(pre_tabu(t,1),pre_tabu(t,2)) = tt;
    end
    Tabu_list(max(swap),min(swap)) = Tabu_list(max(swap),min(swap)) + 1;
    permutation = Best_permutation(I,:);

    if Best_Objective(I) < Best_record(I)
        Best_record(I+1) = Best_Objective(I);
        Best_record_permutation(I+1,:) = Best_permutation(I,:);
    else
        Best_record(I+1) = Best_record(I);
        Best_record_permutation(I+1,:) = Best_record_permutation(I,:);
    end

    % Termination condition : Convergence
    if (Best_record(I+1) - Best_record(I)) == convergence_threshold
        convergence_times = convergence_times + 1;

        if convergence_times > convergence_max_times
            stop = I+1;
            break
        end
    elseif (Best_record(I+1) - Best_record(I)) ~= convergence_threshold
        convergence_times = 0;
    end
end
time = toc;
%% Final Optimization Result
if convergence_times > convergence_max_times
    ans_permutation = Best_record_permutation(stop,:);
    ans_distance = Best_record(stop);
    ite_time = stop - 1;
else
    [mm,ii] = min(Best_record);
    ans_permutation = Best_record_permutation(ii,:);
    ans_distance = Best_record(ii); 
    ite_time = iteration;
end
disp([newline,'Best Permutation = [ ',num2str(ans_permutation),' ]',newline,...
    'Minimum Distance : ',num2str(ans_distance),newline,...
    'Iteration times : ',num2str(ite_time),newline,...
    'Total time spent for Tabu = ',num2str(time),' sec'])
%% Evolution History Plot
if convergence_times > convergence_max_times
    convergence  = Best_record(2:stop);
    Ite = 1:length(convergence);
else
    convergence  = Best_record(2:iteration+1);
    Ite = 1:iteration;
end
figure;
plot(Ite,convergence,'xr-');xlabel('Iterations');ylabel('Fitness');
xlim([0,length(Ite)]);ylim([convergence(end)-20,inf]);title('Evolution History');legend('best record so far');
set(gca,'ygrid','on','Gridalpha',0.4);
%% Global optimal solution
all_permutation = perms(1:node);
for i = 1:size(all_permutation,1)
    all_ans(i) = Obj_dis(all_permutation(i,:),distance);
end
fprintf(2, "Global minimum optimal solution = %s\n", num2str(min(all_ans)));
%% Function (Calculate distance)
function [total_distance] = Obj_dis(permutation,distance)
per = permutation;
total_distance = 0;
for i = 1:length(per)-1
    begin_city = per(i);
    to_city = per(i+1);
    total_distance = total_distance + distance(begin_city,to_city);
end
total_distance = total_distance + distance(per(end),per(1)); 
end