%% 데이터 생성 (단층퍼셉트론, 다층퍼셉트론 개념 이해) - 실험 1
close all;
count = 100;
c1count = count;
c1mean  = [17 10];
c1var   = 1;
c11 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c12 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c1 = [c11; c12];
% figure(1), scatter(c1(:,1), c1(:,2)); hold on;

c2count = count;
c2mean  = [10 11];
c2var   = 1;
c21 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c22 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c2 = [c21; c22];
% figure(1), scatter(c2(:,1), c2(:,2)); hold on;

c3count = count;
c3mean = [10 2];
c3var = 3;
c31 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c32 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c3 = [c31; c32];
% figure(1), scatter(c3(:,1), c3(:,2)); hold on;

c4count = count;
c4mean = [17 5];
c4var = 1;
c41 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c42 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c4 = [c41; c42];
C = [c1; c2; c3; c4];
figure(1), scatter(C(:,1), C(:,2)); hold on;

c5count = count;
c5mean = [12 7];
c5var = 1;
c51 = [c5mean(1)+c5var*randn(c5count,1) c5mean(2)+c5var*randn(c5count,1)];
c52 = [c5mean(1)+c5var*randn(c5count,1) c5mean(2)+c5var*randn(c5count,1)];
c5 = [c51; c52];
figure(1), scatter(c5(:,1), c5(:,2)); hold off;
%% 데이터 목표값 배정
c1 = c1';
c2 = c2';
c3 = c3';
c4 = c4';
c5 = c5';
t1 = repmat(0,1,length(c1));
t2 = repmat(0,1,length(c1));
t3 = repmat(0,1,length(c1));
t4 = repmat(0,1,length(c1));
t5 = repmat(1,1,length(c1));
%% 데이터 병합
C0 = [c1 c2 c3 c4];
C1 = [c5];
T0 = [t1 t2 t3 t4];
T1 = [t5];

%% 데이터 순서섞기
x = [C0 C1];
t = [T0 T1];
shuffle_index = randperm(length(t));
x = x(:,shuffle_index);
t = t(:,shuffle_index);
%% 신경망 학습
% x = x;
% t = t;
% Create a Pattern Recognition Network
hiddenLayerSize = 6;
net = patternnet(hiddenLayerSize);
% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
[net,tr] = train(net,x,t,'useGPU','yes');
% Test the Network
y = net(x);
% View the Network
%view(net)
%% 학습용 그래프랑 인지공간 출력
figure(2), plot(t(1:20),'o'), hold on; plot(y(1:20),'x'); hold off;
xrange = -5:30;
yrange = -5:20;
[X, Y] = meshgrid(xrange,yrange);
index = [X(:) Y(:)]';
out = net(index);
out1 = reshape(out, length(yrange), length(xrange));
% out1 = flipud(out1);
figure(2), imagesc(out1);
set(gca,'YDir','normal');
figure(3), mesh(xrange, yrange, out1);
hold on;
scatter3(x(1,:), x(2,:), y);
hold off;
%% 피팅층 설계 시뮬레이션 - 실험 2
close all; clear all;
period = 2*pi;
x = linspace(0,period*5,1000);
y1 = sin(x);
y2 = 0.7*sin(1.8*x-pi/4);
figure, 
plot(x, y1,'--b', x,y2,'*g');
t1 = (y2>y1);
t2 = (y2>0.2);
t = t1 & t2;
hold on;
plot(x,t);
hold off;

t10 = (t == 1);
figure, 
scatter(y1(t10), y2(t10));
hold on;
scatter(y1(~t10), y2(~t10));
hold off;

%% 데이터 순서섞기 
x = [y1; y2];
t = t;
shuffle_index = randperm(length(t));
shuffle_x = x(:,shuffle_index);
shuffle_t = t(:,shuffle_index);
%% 신경망 학습
trainFcn = 'trainlm';
net = fitnet([2 1], trainFcn);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
% Train the Network
numC = size(shuffle_t(1,:),2);
numC = round(numC * 0.5);
% Train the Network
[net tr] = train(net, (shuffle_x(:,1:numC)), shuffle_t(:,1:numC), 'useGPU','yes');
%% 결과확인
out = net(x);
index = out > 0.5;
x1 = x(:,index);
x2 = x(:,~index);
figure,
plot(x(1,:)); hold on;
plot(x(2,:)); % hold on;
plot(out); hold off;
figure,
scatter(x1(1,:), x1(2,:));hold on;
scatter(x2(1,:), x2(2,:));hold off;

%% 가상 데이터 분류용 신경망 학습 시뮬레이션  - 실험 3
clear all; close all;
count = 3000;
c1count = count;
c1mean  = [5 5];
c1var   = 1.5;
c1 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];

c2count = count;
c2mean  = [10 10];
c2var   = 1.5;
c2 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];

c3count = count;
c3mean  = [15 15];
c3var   = 1.5;
c3 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];

c4count = count;
c4mean  = [20 20];
c4var   = 1.5;
c4 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];

figure,
subplot(2,2,1), 
plot( c1(:,1), c1(:,2), 'ro' ); hold on;
plot( c2(:,1), c2(:,2), 'bo' ); hold on;
plot( c3(:,1), c3(:,2), 'go' ); hold on;
plot( c4(:,1), c4(:,2), 'co' ); hold off;
title('scatter plot'); xlabel('A'); ylabel('B');
subplot(2,2,3), 
histogram(c1(:,1),50); hold on; 
histogram(c2(:,1),50); hold on;
histogram(c3(:,1),50); hold on;
histogram(c4(:,1),50); hold off;
title('x axis histogram'); xlabel('A'); ylabel('count');
subplot(2,2,4), 
histogram(c1(:,2),50); hold on; 
histogram(c2(:,2),50); hold on;
histogram(c3(:,2),50); hold on;
histogram(c4(:,2),50); hold off;
title('y axis histogram'); xlabel('B'); ylabel('count');
X = [c1; c2; c3; c4];
L = [repmat(1,size(c1,1),1); repmat(2,size(c2,1),1); repmat(3,size(c3,1),1); repmat(4,size(c4,1),1)];
[Y, W, lambda] = LDA(X, L);
Y = Y*50;
y1 = Y(1:count,:);
y2 = Y(count+1:2*count,:);
y3 = Y(2*count+1:3*count,:);
y4 = Y(3*count+1:4*count,:);
subplot(2,2,2), 
histogram(y1(:,1),50); hold on; 
histogram(y2(:,1),50); hold on;
histogram(y3(:,1),50); hold on;
histogram(y4(:,1),50); hold off;
title('B=A axis histogram'); xlabel('B=A'); ylabel('count');

ctrain = [  c1' ...
            c2' ...
            c3' ...
            c4' ...
         ];
ctarget = [ repmat([10], 1, count) ...
            repmat([20], 1, count) ...
            repmat([30], 1, count) ...
            repmat([40], 1, count) ...
          ];

size_Trainset = size(ctrain,2);
shuffle_idx = randperm(size_Trainset);
shuffled_Trainset = ctrain(:,shuffle_idx);
shuffled_Target = ctarget(:,shuffle_idx);

%% 
% Create a Fitting Network
% g = gpuDevice(1);
trainFcn = 'trainlm';
netXY_3 = fitnet([50 10 4 1], trainFcn);
netXY_3.divideParam.trainRatio = 70/100;
netXY_3.divideParam.valRatio = 15/100;
netXY_3.divideParam.testRatio = 15/100;
% Train the Network
numC = size(shuffled_Trainset(1,:),2);
numC = round(numC * 1);
% Train the Network
[netXY_3 tr] = train(netXY_3, (shuffled_Trainset(:,1:numC)), shuffled_Target(:,1:numC), 'useGPU','yes');
% save('netXY_3.mat','netXY_3');
%%
x = 0.5:0.5:30;
y = 0.5:0.5:30;
z = [1:5:40];
[X Y] = meshgrid(x, y);
tri = delaunay(X,Y);
Z = netXY_3([X(:) Y(:)]');
out = netXY_3([c1(:,1); c2(:,1); c3(:,1); c4(:,1)]');
figure,
subplot(2,2,1), contour(X, Y, reshape(Z,size(x,2),size(y,2)),z, 'ShowText','on'); hold on;
subplot(2,2,1), plot( c1(:,1), c1(:,2), 'ro' ); hold on;
subplot(2,2,1), plot( c2(:,1), c2(:,2), 'bo' ); hold on;
subplot(2,2,1), plot( c3(:,1), c3(:,2), 'go' ); hold on;
subplot(2,2,1), plot( c4(:,1), c4(:,2), 'co' ); hold on;
% axis([0 inf], [0 inf], [0 inf]);
subplot(2,2,2), trisurf(tri, X, Y, reshape(Z,size(x,2),size(y,2))); % camlight right
% axis tight;
% hold on;
% subplot(2,2,2), plot3([c1(:,1); c2(:,1); c3(:,1)], [c1(:,2); c2(:,2); c3(:,2)], Z,20,20),'.','MarkerSize', 8);
subplot(2,2,3), histogram([c1(:,1); c2(:,1); c3(:,1); c4(:,1)], 100);
subplot(2,2,4), histogram(out,100);