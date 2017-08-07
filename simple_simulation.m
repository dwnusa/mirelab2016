%% 데이터 생성
close all;
count = 100;
c1count = count;
c1mean  = [17 10];
c1var   = 1;
c11 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c12 = [c1mean(1)+c1var*randn(c1count,1) c1mean(2)+c1var*randn(c1count,1)];
c1 = [c11; c12];
figure(1), scatter(c1(:,1), c1(:,2)); hold on;

c2count = count;
c2mean  = [10 11];
c2var   = 1;
c21 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c22 = [c2mean(1)+c2var*randn(c2count,1) c2mean(2)+c2var*randn(c2count,1)];
c2 = [c21; c22];
figure(1), scatter(c2(:,1), c2(:,2)); hold on;

c3count = count;
c3mean = [10 2];
c3var = 3;
c31 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c32 = [c3mean(1)+c3var*randn(c3count,1) c3mean(2)+c3var*randn(c3count,1)];
c3 = [c31; c32];
figure(1), scatter(c3(:,1), c3(:,2)); hold on;

c4count = count;
c4mean = [17 5];
c4var = 1;
c41 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c42 = [c4mean(1)+c4var*randn(c4count,1) c4mean(2)+c4var*randn(c4count,1)];
c4 = [c41; c42];
figure(1), scatter(c4(:,1), c4(:,2)); hold on;

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
[net,tr] = train(net,x,t);
% Test the Network
y = net(x);
% View the Network
%view(net)
%% 학습용 그래프랑 인지공간 출력
figure(2), plot(t(1:20),'o'), hold on; plot(y(1:20),'x'); hold off;
[X, Y] = meshgrid(-5:30,-5:20);
index = [X(:) Y(:)]';
out = net(index);
out1 = reshape(out, length(-5:20), length(-5:30));
% out1 = flipud(out1);
figure(2), imagesc(out1);
set(gca,'YDir','normal');
figure(3), mesh(out1);
return;

%% 가상 데이터 시뮬레이션
clear all;
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
% hiddenLayerSize/2]); 100 50 40 30 20 10
% netXY_3.layers{1}.transferFcn = 'logsig';
% netXY_3.layers{2}.transferFcn = 'logsig';
% netXY_2.layers{3}.transferFcn = 'logsig';
% netXY_2.layers{4}.transferFcn = 'hardlim';
% netXY_2.layers{5}.transferFcn = 'hardlim';
% netXY_2.layers{6}.transferFcn = 'hardlim';
netXY_3.divideParam.trainRatio = 70/100;
netXY_3.divideParam.valRatio = 15/100;
netXY_3.divideParam.testRatio = 15/100;
% Train the Network
numC = size(shuffled_Trainset(1,:),2);
numC = round(numC * 1);
% Train the Network
[netXY_3 tr] = train(netXY_3, (shuffled_Trainset(:,1:numC)), shuffled_Target(:,1:numC), 'useParallel','yes','useGPU','yes');
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