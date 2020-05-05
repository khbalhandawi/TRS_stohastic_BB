close all
clear all
clc

% Start server
model = 'TYPE PRS DEGREE 2';
model = 'TYPE ENSEMBLE METRIC LINV WEIGHTS OPTIM'

system('@echo off && for /r %F in (flag_*) do rm -f "%~nF"');
sgtelib_server_start('TYPE PRS DEGREE 2',true)

% Test if server is ok and ready
sgtelib_server_ping;

% Build data points
N = 2;
M = 1;
P = 5000;
X = rand(P,N);
Z = rand(P,M);

% Feed server
sgtelib_server_newdata(X,Z);

% Prediction points
PXX = 50;
XX = rand(PXX,N);

%Prediction
[ZZ,std,ei,cdf] = sgtelib_server_predict(XX);

% Plot
figure; hold on;
plot3(X(:,1),X(:,2),Z,'*k');
plot3(XX(:,1),XX(:,2),ZZ,'or')

% Prediction points
PXX = 100;
XX = rand(PXX,N);

%Prediction
[ZZ,std,ei,cdf] = sgtelib_server_predict(XX);

% Plot
figure; hold on;
plot3(X(:,1),X(:,2),Z,'*k');
plot3(XX(:,1),XX(:,2),ZZ,'or')

% Stop server
sgtelib_server_stop;