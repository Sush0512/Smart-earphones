[x,Fs]= audioread('am1.wav');
[y,Fs]= audioread('am2.wav');
[z,Fs]= audioread('amb4.wav');

[x1,Fs]= audioread('cd1.wav');
[y1,fs]=audioread('c2.wav');

[x2,Fs]= audioread('silence.wav');
[y2,Fs]= audioread('silence1.wav');

% [x3,Fs]= audioread('horn.wav');
% [y3,Fs]= audioread('horn2.wav');

Features1 = stFeatureExtraction(x, 44100, 0.05  , 0.015);
Features2 = stFeatureExtraction(y, 44100, 0.05  , 0.015);
Features3 = stFeatureExtraction(z, 44100, 0.05  , 0.015);


Features4 = stFeatureExtraction(x1, 44100, 0.05  , 0.015);
Features5 = stFeatureExtraction(y1, 44100, 0.05  , 0.015);

Features6 = stFeatureExtraction(x2, 44100, 0.05  , 0.015);
Features7 = stFeatureExtraction(y2, 44100, 0.05  , 0.015);


% Features8 = stFeatureExtraction(x3, 44100, 0.05  , 0.015);
% Features9 = stFeatureExtraction(y3, 44100, 0.05  , 0.015);

inp= horzcat(Features1,Features2, Features3,Features4, Features5, Features6, Features7);
t1= [ones(1,2600), zeros(1,4667), zeros(1,3306)];
t2= [zeros(1,2600), ones(1,4667), zeros(1,3306) ];
t3= [zeros(1,2600), zeros(1,4667),ones(1,3306)];

t= vertcat(t1,t2, t3);

%%Neural networks training
prnet=newpr(inp,t,30);
trainedprnet=train(prnet,inp,t);






