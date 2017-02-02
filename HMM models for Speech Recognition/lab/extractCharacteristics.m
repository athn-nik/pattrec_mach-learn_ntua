function C = extractCharacteristics(digitaudioname, T, Toverlap, Nc)
    b = [1, -0.97];
    a = [1, 0];
    
    Q = 24;
    f_min = 300;
    f_max = 8000;
    [s_o,fs] = audioread(digitaudioname);
    %fprintf('%d\n', fs);
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
    
    for frame_ii = 1: size(S,1)
            frame = S(frame_ii,:);
            fftframe = fft(frame,nfft); % fft dianusma 256 simeiwn
            fftframe = repmat(fftframe,Q,1);
            y = fftframe .* H;
            E(frame_ii,:) = sum(abs(y).^2,2)/nfft;
            % ------------------------ Step 6 ------------------------ %
            G(frame_ii,:) = log10(E(frame_ii, :));
            % ------------------------ Step 7 ------------------------ %
            C(frame_ii,:) = dct(G(frame_ii, :));
            Ctemp(frame_ii,:) = C(frame_ii, 1:Nc);
            
    end
    C = Ctemp';
end

