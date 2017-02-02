%% Prelab 2

 filelist = dir('./digits2016/*.wav');
% Pre-allocate an array to store some per-file information.
for index = 1 : length(filelist)
    %% Step 1
    
    fprintf('Processing %s\n', filelist(index).name);
    % Read the sample rate Fs and store it.
    [S0, Fs] = audioread(strcat('./digits2016/',filelist(index).name));
    %% Step 2
    %y = filter(b(numerator coeff..s),a(denominator coeff..s),x)
    Sp=filter([1 -0.97],[1 0],S0);
    %% Step 3
    T = 0.025; % duration of each frame
    Toverlap = 0.01; % overlap 
    Samples_per_Frame=T*Fs; % sampling frequency(samples/sec)*duration of frame(sec) = 
    % samples per frame
    noverlap=Toverlap*Fs; % # of overlappping samples between sequential frames
    % buffer return each frame as column!!(so use inverse matrix for ease mult) with p cells padding in the
    % starting frame and sufficient padding in the last frame 
    S = buffer(Sp,Samples_per_Frame,noverlap)';
    %construct hamming window of size samples/frame returns a column vector!
    w = hamming(Samples_per_Frame);
    %replicate frames times hamming to multiply with signal(size (S,1) is # of frames)
    %use inverse for correct mult
    hamming_window = repmat(w', size(S,1), 1); % dimiourgia parathurou hamming
    S = S .* hamming_window;  
    %% Step 4
    
    f_min = 300;
    f_max = 8000;
    Q=24;
    nfft = 2^nextpow2(Samples_per_Frame);   % euresh shmeiwn fft

    k = linspace(f_min,f_max,nfft);

    fc_min = 2595*log10(1+f_min/700);   % min suxnothta sth mel
    fc_max = 2595*log10(1+f_max/700);   % max suxnothta sth mel
    fc = linspace(fc_min,fc_max,Q+2);   % grammikos xwros suxnothtwn sth mel
     fmel = 700*(10.^(fc/2595)-1);
        f = floor((nfft+1)*fmel/Fs);  
    % ftiaxnoume ta f wste na einai idia me kapoia apo tis suxnotites k
%     freq = repmat(flin, nfft, 1)';
%     knew = repmat(k, Q+2, 1);
%     diff = abs(knew-freq);
%     [mindiff, indexes] = min(diff,[],2);
%     f = k(indexes);
%     
%         % Ypologismos filtrou
%         H = zeros(Q, nfft);
%         for m =2:length(fc)-1
%             for p=1:nfft
%                 if (k(p)<f(m-1))
%                     H(m-1,p)=0;
%                 elseif (k(p)>=f(m-1) && k(p)<=(f(m)))
%                     H(m-1,p)=(k(p)-f(m-1))/(f(m)-f(m-1));
%                 elseif (k(p)>=f(m) && k(p)<=f(m+1))
%                     H(m-1,p)=(f(m+1)-k(p))/(f(m+1)-f(m));
%                 elseif (k(p)>f(m+1))
%                     H(m-1,p)=0;
%                 end
%             end
%         end
         H = zeros(Q,nfft);
        for mel_filt_no = 2:Q+1
            
             for k=1:nfft
                 if  (k>=f(mel_filt_no-1) && k<=f(mel_filt_no))
                     H(mel_filt_no-1,k)= (k-f(mel_filt_no-1))/(f(mel_filt_no)-f(mel_filt_no-1));
                 end
                 
                 if    (k>=f(mel_filt_no) && k<=f(mel_filt_no+1))
                     H(mel_filt_no-1,k)= (f(mel_filt_no+1)-k)/(f(mel_filt_no+1)-f(mel_filt_no));
                 end
%             for ii = f(jj-1):f(jj)
%                 H(jj-1,ii) = ((f(jj)-f(jj-1))-(f(jj)-ii))/(f(jj)-f(jj-1));
%             end
%             for ii = f(jj):f(jj+1)
%                 H(jj-1,ii) = 1-((f(jj+1)-f(jj))-(f(jj+1)-ii))/(f(jj+1)-f(jj));
%             end
             end
        end
        
        if index==1
             figure(); hold on;
            for p=1:Q
                plot(k,H(p,:));
            end
            hold off;
        end
        
        
        Nc=13;
        j=str2double(regexp(filelist(index).name,'\d*','Match'));
        %take the number from each name to see the speaker
        for frame = 1: size(S,1) %for all the frames
           
            frame_in_process = S(frame,:); %take each
            fftframe1 = fft(frame_in_process,nfft); % one fft transform
            fftframe = repmat(fftframe1,Q,1); %replicate for passing through 24 filters
            frame_out = fftframe .* H;
            %% Step 5
            E(frame,:) = sum(abs(frame_out).^2,2)/nfft; %energy of output
          
            %% Step 6
            G(frame,:) = log10(E(frame, :));
            %% Step 7
            %take dct of G
            %fix it in the correct line of cell array
             if strncmp(filelist(index).name,'one',3)
                   id=1;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'two',3)
                   id=2;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'three',5)
                    id=3;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'four',4)
                   id=4;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'five',4)
                   id=5;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'six',3)
                   
                   id=6;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'seven',5)
                   id=7;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'eight',5)
               if j==1
                        if frame==20
                            F_20 = 2*abs(fftframe1).^2/nfft;
                            
                            E_20 = E(frame,:);
                            figure(frame);
                            plot(E_20);
                            G_20=  G(frame,:);
                        end
                        if frame==25
                            F_25 = 2*abs(fftframe1).^2/nfft;

                            E_25 = E(frame,:);
                            figure(frame);
                            plot(E_25);
                            G_25 = G(frame,:);
                        end
                 end
                   id=8;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             elseif strncmp(filelist(index).name,'nine',4)
                   id=9;
                   C{id,j}(frame,:) = dct(G(frame, :));
                   C_keep_only13{id,j}(frame,:) = C{id,j}(frame,1:Nc);
                   
             end
            
        end
end

%Athanasiou Nikolaos 03112074
k1=4; k2=7;
n1=mod(27,Nc);
n2=mod(4,Nc);


%histogram of coefficients
figure(1);
subplot(2,1,1); hist(C_keep_only13{4,1}(:,n1)); title('Coefficients Ci for digit 4 n1=1');
subplot(2,1,2); hist(C_keep_only13{4,1}(:,n2)); title('Coefficients Ci for digit 4 n2=4');

figure(2);
subplot(2,1,1); hist(C_keep_only13{7,1}(:,n1)); title('Coefficients Ci for digit 7 n1=1');
subplot(2,1,2); hist(C_keep_only13{7,1}(:,n2)); title('Coefficients Ci for digit 7 n2=4');
%% Step 8
E_8_1_20_reconstructed = 10.^idct(C_keep_only13{8,1}(20,:),Q);
E_8_1_25_reconstructed = 10.^idct(C_keep_only13{8,1}(25,:),Q);

Erec_20 = zeros(1,nfft/2);
Erec_25 = zeros(1,nfft/2);
%fix the reconstructed frames
for idx = 1:Q
    Erec_20(f(idx)) = E_8_1_20_reconstructed(idx);
    Erec_25(f(idx)) = E_8_1_25_reconstructed(idx);
end
%plot my reconstruction
figure('Name','Energy of frame 20 of digit 8 speaker 1');
hold on;
plot(F_20(1:nfft/2));
plot(Erec_20,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;

figure('Name','Energy of frame 25 of digit 8 speaker 1');
hold on; 
plot(F_25(1:nfft/2));
plot(Erec_25,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;
%% Step 9
symbols = {'.' 'o' 'x' '+' '*' 's' 'd' 'v' '>'};
colors={'b' 'y' 'c' 'r' 'g' 'm' [ 0.3373    0.1333    0.0039] 'k' ([241 32 155] ./ 255)};
figure('Name','C MeanValues','NumberTitle','off');
MeanData = cell(9,15);
MeanDigit = cell(9,1);
%plot them different color different symbol
for i = 1:9
    MeanDigit{i} = zeros(1,Nc);
    for j = 1:15
        if((i == 8 && j == 7)||(i == 6 && j == 12))
            continue
        end
        MeanData{i,j} = mean(C_keep_only13{i,j},1);
        MeanDigit{i} = MeanDigit{i} + MeanData{i,j};
        
         if i==7 
             fig=plot(MeanData{i,j}(n1),MeanData{i,j}(n2),symbols{i}); 
             set(fig(1),'color',[ 0.3373    0.1333    0.0039]);

         elseif i==9
              fig =plot(MeanData{i,j}(n1),MeanData{i,j}(n2),symbols{i}); 
              set(fig(1),'color',([241 32 155] ./ 255));

         else
             plot(MeanData{i,j}(n1),MeanData{i,j}(n2),[symbols{i},colors{i}]); 
             hold on;
         end
    end
end

for idx=1:9
    MeanDigit{idx}=MeanDigit{idx}/15;
    if(idx== 6 && idx== 8)
            MeanDigit{idx} = 15*MeanDigit{idx}/14;
    end
end
xlabel(sprintf('Mean C(with n1=%d)',n1));
ylabel(sprintf('Mean C(with n2=%d)',n2));

for i=1:9
%plot these means bold and big
    if i==7 
       h(i)=plot(MeanDigit{i}(n1),MeanDigit{i}(n2),symbols{i},'MarkerSize', 10,'Linewidth',3);
       set(h(1),'color',[ 0.3373    0.1333    0.0039]);

    elseif i==9
       h(i)=plot(MeanDigit{i}(n1),MeanDigit{i}(n2),symbols{i},'MarkerSize', 10,'Linewidth',3);
       h(1) =plot(MeanData{i,j}(n1),MeanData{i,j}(n2),symbols{i}); 
                       set(h(1),'color',([241 32 155] ./ 255));
    else
       h(i)=plot(MeanDigit{i}(n1),MeanDigit{i}(n2),[symbols{i},colors{i}],'MarkerSize', 10,'Linewidth',3);
    
    end
    hold on;
end

digitnames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'};
legend(h, digitnames);
