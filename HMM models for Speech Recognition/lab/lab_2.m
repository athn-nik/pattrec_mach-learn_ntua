%% Lab 2 Pattern Recognition
close all;


addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools
load('C.mat');
 %C_keep_only13=Ctemp;
C_keep_only13 = cellfun(@transpose,C_keep_only13,'UniformOutput',false);

%% Step 11
Nm = 2; % # gaussians
Ns = 5; % # states
Nc=13;%num of coefficients for each signal <=> features

train_ratio=13;
%70% of data used for train
for digit = 1:9 % for every digit create a hmm
    %% Step 10
    % initial prior probability
    %0-> i<>1
    %1-> i=1
    prior{digit} = zeros(Ns,1);
    prior{digit}(1) = 1;

    
    % left-right hmm1 transition matrix
    % with 
    transitions{digit} = rand(Ns,Ns);
    for i = 1:Ns
        for j = 1:Ns
            if (i>j)
                transitions{digit}(i,j) = 0; 
            end
            if (i+1<j)
                transitions{digit}(i,j) = 0; 
            end
        end
    end
    transitions{digit} = mk_stochastic(transitions{digit});
    % initialize mixture of gauusians    
    % mixture of gauss (kmeans is not needed is set by default)
    [mean_val{digit}, variance{digit}] = mixgauss_init(Ns*Nm, [C_keep_only13{digit,1:train_ratio}], 'full');
    mean_val{digit} = reshape(mean_val{digit}, [Nc Ns Nm]);
    variance{digit} = reshape(variance{digit}, [Nc Nc Ns Nm]);
    mixmat{digit} = mk_stochastic(rand(Ns,Nm));
     % fix data for omitted speakers
    if(digit == 8) 
        mhmm_em_data = C_keep_only13(digit,[1:6 8:train_ratio]);
     elseif (digit ==6)
         mhmm_em_data = C_keep_only13(digit,[1:11 13:train_ratio ]);
     else
        mhmm_em_data = C_keep_only13(digit,1:train_ratio);
    end
    
    % improve estimates using EM
    [LL{digit}, prior_trained{digit}, transitions_trained{digit}, mean_val_trained{digit},...
    variance_trained{digit}, mixmat_trained{digit}] = mhmm_em(mhmm_em_data, prior{digit},...
    transitions{digit}, mean_val{digit}, variance{digit}, mixmat{digit},'max_iter', 10,'verbose', 0);
    
end
%% Step 12
%test my data
B=cell(9,4);
for i=1:9
     for j=12:15
       B{i,j-11}=C_keep_only13{i,j};
     end
end

for i=1:9
    for j=12:15
        if ((i==6)&&(j==12)) 
            continue;
        end
%---------------Extraction of Characteristics(from prelab)--------------%
% IT IS NOT NEEDED BECAUSE I HAVE SAVED & LOADED C_keep_only13 Matrix from
% prelab!!!
%            restoredefaultpath
%            rehash toolboxcache
%            savepath
%           digitaudioname = sprintf('./digits2016/%s%d.wav', strjoin(digitnames(i)), j);
%             = extractCharacteristics(digitaudioname, 0.025,0.01,13);
%     b = [1, -0.97];
%     a = [1, 0];
%     T=0.025;
%     Toverlap=0.1;
%     Nc=13;
%     Q = 24;
%     f_min = 300;
%     f_max = 8000;
%     [s_o,fs] = audioread(digitaudioname);
%     %fprintf('%d\n', fs);
%     s_p = filter(b,a,s_o);
%     % ------------------------ Step 3 ------------------------ %
%     n = fs*T;               % deigmata ana plaisio
%     noverlap = fs*Toverlap;      % epikaluptomena deigmata
%     S = buffer(s_p,n,noverlap)';    % diaxwrismos se plaisia
%     hamming_window = repmat(hamming(n)', size(S,1), 1); % dimiourgia parathurou hamming
%     S = S .* hamming_window;    % parathurwsi
%     nfft = 2^nextpow2(n);   % euresh shmeiwn fft
%     % ------------------------ Step 4 ------------------------ %
%     k = linspace(f_min,f_max,nfft);
%     fc_min = 2595*log10(1+f_min/700);   % min suxnothta sth mel
%     fc_max = 2595*log10(1+f_max/700);   % max suxnothta sth mel
%     fc = linspace(fc_min,fc_max,Q+2);   % grammikos xwros suxnothtwn sth mel
%     fmel = 700*(10.^(fc/2595)-1);
%     f = floor((nfft+1)*fmel/fs);        % antistoixish syxnothtwn sta fft bins    
% 
%     % Ypologismos filtrou
%     
%          H = zeros(Q,nfft);
%         for mel_filt_no = 2:Q+1
%             
%              for k=1:nfft
%                  if  (k>=f(mel_filt_no-1) && k<=f(mel_filt_no))
%                      H(mel_filt_no-1,k)= (k-f(mel_filt_no-1))/(f(mel_filt_no)-f(mel_filt_no-1));
%                  end
%                  
%                  if    (k>=f(mel_filt_no) && k<=f(mel_filt_no+1))
%                      H(mel_filt_no-1,k)= (f(mel_filt_no+1)-k)/(f(mel_filt_no+1)-f(mel_filt_no));
%                  end
%        end
%     for frame_ii = 1: size(S,1)
%             frame = S(frame_ii,:);
%             fftframe = fft(frame,nfft); % fft dianusma 256 simeiwn
%             fftframe = repmat(fftframe,Q,1);
%             y = fftframe .* H;
%             E(frame_ii,:) = sum(abs(y).^2,2)/nfft;
%             % ------------------------ Step 6 ------------------------ %
%             G(frame_ii,:) = log10(E(frame_ii, :));
%             % ------------------------ Step 7 ------------------------ %
%             C(frame_ii,:) = dct(G(frame_ii, :));
%             Ctemp(frame_ii,:) = C(frame_ii, 1:Nc);
%             
%     end
%     B{i,j-11} = Ctemp';
%     addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools

        for digit = 1:9
            likelihood(digit) = mhmm_logprob(B{i,j-11}, prior_trained{digit}, transitions_trained{digit},...
                mean_val_trained{digit}, variance_trained{digit}, mixmat_trained{digit});
        end
        % find where is classified
        [~, hmm_digit_classifier(i,j-11)] = max(likelihood); 
    end
end
%% Step 13
% plot log likelihood
k1=mod(03112074,10);
figure('Name','Learning Curve for digit: 4','NumberTitle','off');
plot(LL{k1},'*r--');
grid on; title('Learning Curve for digit: 4');
xlabel('#Iterations');
ylabel('LogLikelihood');


%% Step 14
%calculate confusion matrix
ConfusionMatrix = zeros(9,9);
for i=1:9
    for j=1:9
        
        ConfusionMatrix(i,j) = size(find(hmm_digit_classifier(i,:)==j),2);
    end
end

figure('Name','Confusion Matrix','NumberTitle','off');
speakers = 4;
image(ConfusionMatrix);
colormap(autumn(speakers));
xlabel('#Classified Utterances');
ylabel('Digit Utterances');

SuccessRate = trace(ConfusionMatrix)/sum(sum(ConfusionMatrix));

%% Step 15
for i = 1:9 % for all test digits
    
    figure('Name', sprintf('Viterbi for digit %d', i));
    
    for j = 1:4 % for all testing utterances
        if ((i==6)&&(j==1))
            continue;
        end
        % B(i,t) = Pr(y(t) | Q(t)=i),Y is observation
        
        
        Vit{i,j} = mixgauss_prob(B{i,j}, mean_val_trained{i}, variance_trained{i}, mixmat_trained{i});
        % Viterbi path(t) = q(t)
        vpath{i,j} = viterbi_path(prior_trained{i}, transitions_trained{i}, Vit{i,j});
        %plot viterbi path different color-marker for each speaker 
        switch j
            case 1 

                plot(vpath{i,j},'color', 'r',...
                    'linestyle','--','marker','+');
                title(['Viterbi: ',num2str(i)]);
                hold on;
            case 2

                plot(vpath{i,j},'color', 'g',...
                    'linestyle','--','marker','o');
                title(['Viterbi: ',num2str(i)]);
                hold on;
            case 3

                plot(vpath{i,j},'color','b',...
                    'linestyle','--','marker','p');
                title(['Viterbi: ',num2str(i)]);
                hold on;
            case 4
                plot(vpath{i,j},'color','y',...
                    'linestyle','--','marker','x');
                title(['Viterbi: ',num2str(i)]);
                hold on;
        end
        
    end
end