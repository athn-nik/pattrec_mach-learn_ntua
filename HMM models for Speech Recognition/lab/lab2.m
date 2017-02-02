clear all;
close all;

addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools
load('c1.mat');
% ---- Step 11 ----%
Nc = 13;
Ns = 5;
Nm = 2;
pi = cell(1,9);
A = cell(1,9);
temp1 = zeros(Ns);
temp2 = eye(Ns);
temp1(:,2:end) = temp2(:,1:end-1);
temp1 = temp2 + temp1;
speakers = 13;
for digit=1:9
    pi{digit} = zeros(Ns,1);
    pi{digit}(1) = 1;
    A{digit} = rand(Ns, Ns);
    A{digit} = A{digit}.*temp1;
    A{digit} = mk_stochastic(A{digit});
    
    [mean{digit}, variance{digit}] = mixgauss_init(Ns*Nm, [Ctemp{digit,1:speakers}], 'full');
    mean{digit} = reshape(mean{digit}, [Nc Ns Nm]);
    variance{digit} = reshape(variance{digit}, [Nc Nc Ns Nm]);
    mixmat{digit} = mk_stochastic(rand(Ns, Nm));
    
    if (digit == 8)
        data = Ctemp(digit, [1:6 8:speakers]); % eksairoume to eight7
    elseif digit==6
        data = Ctemp(digit, [1:11 13:speakers]); % eksairoume to six12
    else
        data = Ctemp(digit,1:speakers);
    end
    
    [loglik{digit}, pi_train{digit}, A_train{digit}, mean_train{digit}, ...
        variance_train{digit}, mixmat_train{digit}] = mhmm_em(data, pi{digit}, ...
        A{digit}, mean{digit}, variance{digit}, mixmat{digit}, 'max_iter', 10, 'verbose', 0);
end

figure('Name','Learning Curve for digit: 4','NumberTitle','off');
plot(loglik{4},'*r--');
grid on; title('Learning Curve for digit: 4');
xlabel('#Iterations');
ylabel('LogLikelihood');
digitnames = {'one', 'two', 'three', 'four', 'five', 'six', 'seven',...
                'eight', 'nine'};

for i=1:9
    for j=12:15
        if ((i==6)&&(j==12)) 
            continue;
        end
        restoredefaultpath
        rehash toolboxcache
        savepath
        digitaudioname = sprintf('./digits2016/%s%d.wav', strjoin(digitnames(i)), j);
        C{i,j-11} = extractCharacteristics(digitaudioname, 0.025,0.01,13);
        addpath ./HMM ./KPMtools ./KPMstats ./Netlab ./prtools

        for digit = 1:9
            likelihood(digit) = mhmm_logprob(C{i,j-11}, pi_train{digit}, A_train{digit},...
                mean_train{digit}, variance_train{digit}, mixmat_train{digit});
        end
        [~, whichDigit(i,j-11)] = max(likelihood); 
    end
end

hmmClass = whichDigit;
k1 = 5;
figure();
plot(loglik{k1});


ConfusionMatrix = zeros(9,9);
for i=1:9
    for j=1:9
        ConfusionMatrix(i,j) = size(find(hmmClass(i,:)==j),2);
    end
end

figure('Name','Confusion Matrix','NumberTitle','off');
speakers = 4;
image(ConfusionMatrix);
colormap(bone(speakers));
xlabel('#Classified Utterances');
ylabel('Digit Utterances');

SuccessRate = trace(ConfusionMatrix)/sum(sum(ConfusionMatrix));
text(0,0,['SuccessRate = ',num2str(SuccessRate*100),'%']);

figure('Name','Most Probable Sequence(Viterbi)','NumberTitle','off');
chroma = { };

for i = 1:9 % for all test digits
    
    figure('Name', sprintf('Viterbi for digit %d', i));
    
    for j = 1:speakers % for all testing utterances
        if ((i==6)&&(j==1))
            continue;
        end
        % B(i,t) = Pr(y(t) | Q(t)=i),Y is observation
        
        
        B{i,j} = mixgauss_prob(C{i,j}, mean_train{i}, variance_train{i}, mixmat_train{i});
        % Viterbi path(t) = q(t)
        vpath{i,j} = viterbi_path(pi_train{i}, A_train{i}, B{i,j});
                
        red = 0.2 + 0.4* j/speakers;
        green = 0.2* j/speakers;
        blue = 0.8 - 0.6* j/speakers;
        plot(vpath{i,j},'color', [red green blue],...
            'linestyle','--','marker','*');
        title(['Viterbi: ',num2str(i)]);
        hold on;
        grid on;
        
    end
end