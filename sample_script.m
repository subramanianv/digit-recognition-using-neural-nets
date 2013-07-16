I=imread('5.png');
J=rgb2gray(I);
K=imcomplement(J); %the digit has to be in white color and the background has to be in black
L=imresize(K,[28 28]);
O=reshape(L,1,28*28);
Q = double(O) ./ 255 ;
params=load('params.mat'); % contains the weights of the neural net
Theta1 = params.Theta1 ;
Theta2 = params.Theta2 ;
predict1(Theta1,Theta2,Q)