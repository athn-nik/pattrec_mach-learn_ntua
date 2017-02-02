clear all;
close all;


digitnames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven',...
                'eight', 'nine'};
b = [1, -0.97];
a = [1, 0];
T = 0.025;
Toverlap = 0.01;
Q = 24;
f_min = 300;
f_max = 8000;

speakers = 15;
for i = 1:9
    for j = 1:speakers
        if (((i==6)&&(j==12))||((i==8)&&(j==7))) 
            continue;
        end
        % ------------------------ Step 1 ------------------------ %
        clear s_o s_p S freq hamming_window H E G 
        digitaudioname = sprintf('./digits2016/%s%d.wav', strjoin(digitnames(i)), j)
        [s_o,fs] = audioread(digitaudioname);
        % ------------------------ Step 2 ------------------------ %
        s_p = filter(b,a,s_o);
        % ------------------------ Step 3 ------------------------ %
        n = fs*T;               % deigmata ana plaisio
        noverlap = fs*Toverlap;      % epikaluptomena deigmata
        S = buffer(s_p,n,noverlap)';    % diaxwrismos se plaisia
        hamming_window = repmat(hamming(n)', size(S,1), 1); % dimiourgia parathurou hamming
        S = S .* hamming_window;    % parathurwsi
        nfft = 2^nextpow2(n);   % euresh shmeiwn fft
        % ------------------------ Step 4 ------------------------ %
        k = linspace(f_min,f_max,nfft);
        fc_min = 2595*log10(1+f_min/700);   % min suxnothta sth mel
        fc_max = 2595*log10(1+f_max/700);   % max suxnothta sth mel
        fc = linspace(fc_min,fc_max,Q+2);   % grammikos xwros suxnothtwn sth mel
        fmel = 700*(10.^(fc/2595)-1);
        f = floor((nfft+1)*fmel/fs);        % antistoixish syxnothtwn sta fft bins    
    
        % Ypologismos filtrou
        H = zeros(Q,nfft);
        for jj = 2:Q+1
            for ii = f(jj-1):f(jj)
                H(jj-1,ii) = ((f(jj)-f(jj-1))-(f(jj)-ii))/(f(jj)-f(jj-1));
            end
            for ii = f(jj):f(jj+1)
                H(jj-1,ii) = 1-((f(jj+1)-f(jj))-(f(jj+1)-ii))/(f(jj+1)-f(jj));
            end
        end
        
        if ((i == 1)&&(j == 1))
            % ------------------------ Step 5 ------------------------ %    
            frame_i = 10;
            frame = S(frame_i,:);
            fftframe = fft(frame,nfft); % fft dianusma 256 simeiwn
            F_1_1_35 = 2*abs(fftframe).^2/nfft;
            fftframe = repmat(fftframe,Q,1);
            y = fftframe .* H;
            E_1_1_35 = sum(abs(y).^2,2)/nfft;
            G_1_1_35 = log10(E_1_1_35);
            figure('Name',sprintf('Energy of frame %d',frame_i));
            plot(E_1_1_35);
            print -djpeg 'frame_i_original.jpg'

            frame_j = 20;
            frame = S(frame_j,:);
            fftframe = fft(frame,nfft); % fft dianusma 256 simeiwn
            F_1_1_40 = 2*abs(fftframe).^2/nfft;
            fftframe = repmat(fftframe,Q,1);
            y = fftframe .* H;
            E_1_1_40 = sum(abs(y).^2,2)/nfft;
            G_1_1_40 = log10(E_1_1_40);
            figure('Name',sprintf('Energy of frame %d',frame_j));
            plot(E_1_1_40);
            print -djpeg 'frame_j_original.jpg'
        end
        
        Nc = 13;
        for frame_ii = 1: size(S,1)
            frame = S(frame_ii,:);
            fftframe = fft(frame,nfft); % fft dianusma 256 simeiwn
            fftframe = repmat(fftframe,Q,1);
            y = fftframe .* H;
            E(frame_ii,:) = sum(abs(y).^2,2)/nfft;
            % ------------------------ Step 6 ------------------------ %
            G(frame_ii,:) = log10(E(frame_ii, :));
            % ------------------------ Step 7 ------------------------ %
            C{i,j}(frame_ii,:) = dct(G(frame_ii, :));
            Ctemp{i,j}(frame_ii,:) = C{i,j}(frame_ii, 1:Nc);
        end
        Ctemp{i,j} = (Ctemp{i,j})';
    end
end

save('C_features.mat', 'Ctemp');
%%
% ------------------------ Step 7b ------------------------ %
k1 = 5; k2 = 6; n1 = mod(16,Nc); n2 = mod(5,Nc);  % Provatas 03111065
k1 = 6; k2 = 5; n1 = mod(10,Nc); n2 = mod(6,Nc);  % Karamanolakis 03111006

figure('Name','Histogramms','NumberTitle','off');

subplot(2,2,1); hist(Ctemp{k1,1}(:,n1)); title(sprintf('C(%d),%s', n1, strjoin(digitnames(k1))));
subplot(2,2,2); hist(Ctemp{k1,1}(:,n2)); title(sprintf('C(%d),%s', n2, strjoin(digitnames(k1))));
subplot(2,2,3); hist(Ctemp{k2,1}(:,n1)); title(sprintf('C(%d),%s', n1, strjoin(digitnames(k2))));
subplot(2,2,4); hist(Ctemp{k2,1}(:,n2)); title(sprintf('C(%d),%s', n2, strjoin(digitnames(k2))));
print -djpeg 'histogram.jpg'

% ------------------------ Step 8 ------------------------ %
E_1_1_35_reconstructed = 10.^idct(Ctemp{1,1}(frame_i,:),Q);
E_1_1_40_reconstructed = 10.^idct(Ctemp{1,1}(frame_j,:),Q);

Erec_35 = zeros(1,nfft/2);
Erec_40 = zeros(1,nfft/2);
for kk = 1:Q
    Erec_35(f(kk)) = E_1_1_35_reconstructed(kk);
    Erec_40(f(kk)) = E_1_1_40_reconstructed(kk);
end
figure('Name',sprintf('Energy of frame %d',frame_i));
hold on; grid on;
plot(F_1_1_35(1:nfft/2));
plot(Erec_35,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;
print -djpeg 'frame_i_reconstructed.jpg'

figure('Name',sprintf('Energy of frame %d',frame_j));
hold on; grid on;
plot(F_1_1_40(1:nfft/2));
plot(Erec_40,'r');
legend('Power Spectrum', 'Reconstructed Energy');
hold off;
print -djpeg 'frame_j_reconstructed.jpg'
% ------------------------ Step 9 ------------------------ %
symbols = {'.' 'o' 'x' '+' '*' 's' 'd' 'v' 'p'};

figure('Name','C MeanValues','NumberTitle','off');

MeanData = cell(9,15);
MeanDigit = cell(9,1);
for i = 1:9
    MeanDigit{i} = zeros(1,Nc);
    for j = 1:15
        
        % exclude eight7 & six12
        if((i == 8 && j == 7)||(i == 6 && j == 12))
            continue
        end
        MeanData{i,j} = mean(Ctemp{i,j},1);
        MeanDigit{i} = MeanDigit{i} + MeanData{i,j};
        plot(MeanData{i,j}(n1),MeanData{i,j}(n2),[symbols{i},'g']); 
        grid on; hold on;
    end
    if(i~= 6 && i~= 8)
        MeanDigit{i} = MeanDigit{i}/15;
    else
        MeanDigit{i} = MeanDigit{i}/14;
    end
end

xlabel(sprintf('MeanValue C(n1=%d)',n1));
ylabel(sprintf('MeanValue C(n2=%d)',n2));

for i=1:9
    h(i)=plot(MeanDigit{i}(n1),MeanDigit{i}(n2),[symbols{i},'r']);
    hold on;
    grid on;
end
legend(h, digitnames);
print -djpeg 'meanDiagram.jpg'
