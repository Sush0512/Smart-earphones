
win=0.05;
 step= 0.015;
 %r = audiorecorder(44100, 16, 1);
while(1)
    %r=audioRecorderOnline();
    
    r = audiorecorder(44100, 16, 1);
    record(r,5);

    t=cputime;
    while(cputime-t<10)
    end
    stop(r); 
    a=getaudiodata(r)% speak into microphone...
    %audiowrite('test1.wav',a,44100);
    b=stFeatureExtraction(a, 44100, 0.05  , 0.015);
    output = sim(trainedprnet, b)
    o1=0;
    o2=0;
    o3=0;
  [row,col]= size(b)
    for i=1:col
        o1= o1+ output(1,i);
        o2= o2+ output(2,i);
        o3= o3+ output(3,i);
        
    end
    o1= o1/length(b)
    o2= o2/length(b)
    o3= o3/length(b)
   
    %If the test belongs to audio class
    if(o1>o2 && o1>o3 && o1>0.6) %Ambulance
        SoundVolume(.00);
    
    else 
       SoundVolume(1); 
    
    end
    
end