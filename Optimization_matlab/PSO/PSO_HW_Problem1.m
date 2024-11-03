clc;clear;close all;
%% Tuning Parameters
x_upper = 1; x_lower = -1;          % x upper bound and lower bound
y_upper = 1; y_lower = -2;          % y upper bound and lower bound
xy_upper = 1;                       % x+y upper bound
particle = 800;                     % swarm size
c1 = 3; c2 = 3;                     % Acceleration constants : Full model (c1,c2 > 0)
mv = 0.5;                           % Velocity limit
x_v_max = mv*(x_upper-x_lower);     % x_vmax = mv * x_range
y_v_max = mv*(x_upper-x_lower);     % y_vmax = mv * y_range
w_upper = 1; w_lower = 0;           % Inertia weight upper bound and lower bound
Iteration = 50;                     % Iteration 迭代次數
Best_sol = zeros(1,Iteration);      % 用來紀錄每次迭代得出的最佳解
Best_x = zeros(1,Iteration);        % 用來紀錄每次迭代得出的最佳解對應的x
Best_y = zeros(1,Iteration);        % 用來紀錄每次迭代得出的最佳解對應的y
convergence_threshold = 1e-4;       % Termination condition : Convergence
convergence_max_times = 7;          % 如果Fitness值前後相差不到0.0001的次數達到7次，則達到收斂條件，停止迭代
%% Initialize location and velocity
x_vel_range = ((x_upper - x_lower)/2)/2;
x_vel_upper = x_vel_range; x_vel_lower = x_vel_range*(-1);
y_vel_range = ((y_upper - y_lower)/2)/2;
y_vel_upper = y_vel_range; y_vel_lower = y_vel_range*(-1);

tic;
i = 1;
while i <= particle
    x(i) = rand*(x_upper - x_lower) + x_lower;
    y(i) = rand*(y_upper - y_lower) + y_lower;
    Fitness(i) = exp(-0.1*(x(i)^2 + y(i)^2)) + exp(cos(4*pi*x(i)) + cos(2*pi*y(i)));
    x_vel(i) = rand*(x_vel_upper - x_vel_lower) + x_vel_lower;
    y_vel(i) = rand*(y_vel_upper - y_vel_lower) + y_vel_lower;
    % Constraint : Pre-censoring
    if (x(i) + y(i)) > xy_upper
        i = i;
    else
        i = i+1;
    end
end
%% First pBest gBest
P = [x' y' Fitness'];
pBest = P;
[M,I] = max(pBest(:,3));
gBest = [pBest(I,1) pBest(I,2) pBest(I,3)];
clear M I
%% Start Iteration
Best_record = zeros(1,Iteration+1);
convergence_times = 0;
for I = 1:Iteration
    pBest_pre = pBest;
    gBest_pre = gBest;
    P_pre = P;
    x_vel_pre = x_vel;
    y_vel_pre = y_vel;
    x_pre = x;
    y_pre = y;
    % Inertia weight : vary linearly from 1 to 0
    w(I) = w_upper - (I/Iteration)*(w_upper-w_lower);

    for i = 1:particle
        disp("Time: "+ datestr(now,'HH:MM:SS.FFF') + "; Iteration: " + num2str(I) + "; particle " + num2str(i))
        r1 = rand; r2 = rand;
        x_vel(i) = w(I)*x_vel_pre(i) + r1*c1*(pBest_pre(i,1)-P_pre(i,1)) + r2*c2*(gBest_pre(1)-P_pre(i,1));
        y_vel(i) = w(I)*y_vel_pre(i) + r1*c1*(pBest_pre(i,2)-P_pre(i,2)) + r2*c2*(gBest_pre(2)-P_pre(i,2));
        % Damping limit for velocity
        if abs(x_vel(i)) > x_v_max
            x_vel(i) = (x_vel(i)/abs(x_vel(i)))*x_v_max;
        elseif abs(y_vel(i)) > y_v_max
            y_vel(i) = (y_vel(i)/abs(y_vel(i)))*y_v_max;
        end

        x(i) = x_pre(i) + x_vel(i);
        y(i) = y_pre(i) + y_vel(i);
        % Constraint : Repair infeasible location
        if x(i) > x_upper
            x(i) = x_upper;
        elseif x(i) < x_lower
            x(i) = x_lower;
        elseif y(i) > y_upper
            y(i) = y_upper;
        elseif y(i) < y_lower
            y(i) = y_lower;
        elseif (x(i) + y(i)) > xy_upper
            gap = ((x(i) + y(i)) - xy_upper)/2;
            x(i) = x(i) - gap;
            y(i) = y(i) - gap;
        end

        Fitness(i) = exp(-0.1*(x(i)^2 + y(i)^2)) + exp(cos(4*pi*x(i)) + cos(2*pi*y(i)));
    end
    P = [x' y' Fitness'];

    for i = 1:particle
        if P(i,3) > pBest_pre(i,3)
            pBest(i,1) = P(i,1);pBest(i,2) = P(i,2);pBest(i,3) = P(i,3);
        else
            pBest(i,1) = pBest_pre(i,1);pBest(i,2) = pBest_pre(i,2);pBest(i,3) = pBest_pre(i,3);
        end
    end
    
    [PM,PI] = max(pBest(:,3));
    gBest = [pBest(PI,1) pBest(PI,2) pBest(PI,3)];

    [FM,FI] = max(P(:,3));
    if gBest > FM
        best_sol = gBest;
    else
        best_sol = [P(FI,1) P(FI,2) P(FI,3)];
    end
    Best_x(I) = best_sol(1);
    Best_y(I) = best_sol(2);
    Best_Fitness(I) = best_sol(3);

    if Best_Fitness(I) > Best_record(I)
        Best_record(I+1) = Best_Fitness(I);
    else
        Best_record(I+1) = Best_record(I);
    end
    % Termination condition : Convergence
    if (Best_record(I+1) - Best_record(I)) < convergence_threshold
        convergence_times = convergence_times + 1;
        if convergence_times > convergence_max_times
            stop = I+1;
            break
        end
    end
end
time = toc;
%% Optimal solution
[BM,BI] = max(Best_Fitness);
Best_Variable_x = Best_x(BI);
Best_Variable_y = Best_y(BI);
maxF = BM;
disp([newline,'Best Variable x : ',num2str(Best_Variable_x),newline,...
    'Best Variable y : ',num2str(Best_Variable_y),newline,...
    'Maximum of Objective Function : ',num2str(maxF),newline,...
    'Iteration times : ',num2str(stop-1),newline,...
    'Total time spent for PSO = ',num2str(time),' sec',])
%% Evolution History Plot
if convergence_times > convergence_max_times
    convergence  = Best_record(2:stop);
    Ite = 1:length(convergence);
else
    convergence  = Best_record(2:Iteration+1);
    Ite = 1:Iteration;
end
figure;
plot(Ite,convergence,'xr-');xlabel('Iterations');ylabel('Fitness');
xlim([0 length(Ite)]);ylim([8.2 8.4]);title('Evolution History');
legend('best record so far');
set(gca,'YTick',8:0.01:8.5);
set(gca,'ygrid','on','Gridalpha',0.4);