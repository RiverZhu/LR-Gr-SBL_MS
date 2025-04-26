clear;
close all;
clc;
rng('default')


load('ratings_100k')
X_true = spconvert(ratings);



X_true = full(X_true);
[m,n] = size(X_true);
mn = m*n;

idx_nonz = X_true ~= 0;

figure
histogram(X_true(idx_nonz), 'BinEdges', -0.5:5.5, 'Normalization', 'count');


X_max = max(max(X_true));
X_min = 1;


p = 0.4;

index = (X_true~=0);
for i = 1:5
    pp(i) = sum(X_true(index)==i)/length(X_true(index));
end
tau = zeros(4,1);
for i = 1:4
    tau(i) = norminv(sum(pp(1:i)),0,0.431);
end


% tau = [1.5;2.5;3.5;4.5];
% tau = [-0.67; -0.4; -0.06; 0.34];
wvar = 0.1;

MC = 100;
RMSE = nan(MC, 1);
NMAE = nan(MC, 1);




for mc = 1:MC
    fprintf('%d\n', mc)
    rng(mc)
    index = (X_true~=0);
    index_train = zeros(m,n);
    index_test = zeros(m,n);
    
    % 每次随机生成一个P
    for i = 1:m
        index_i = find(index(i,:)==1);
        Omega = index_i(randperm(length(index_i)));
        pmn = p*length(index_i);
        index_train(i,Omega(1:round(pmn))) = 1;
        index_test(i,Omega(round(pmn)+1:end)) = 1;
    end
    P = index_train;


    Y = P.*X_true;
    yy = Y(:);

    r_init = 5;
    r0 = min([r_init,round(min([m,n])/6)]);

    T_max = 50;
    z_A_ext_mat = 0*ones(m,n);
    v_A_ext_mat = 1e1*ones(m,n);
    var_min = 1e-4;
    var_max = 1e3;

    t_lower = zeros(size(yy));
    t_upper = zeros(size(yy));
    Quantize_stepsize = tau(2)-tau(1);
    t_lower(yy == 1) = -inf;
    t_upper(yy == 1) = tau(1);
    t_lower(yy == 2) = tau(1);
    t_upper(yy == 2) = tau(2);
    t_lower(yy == 3) = tau(2);
    t_upper(yy == 3) = tau(3);
    t_lower(yy == 4) = tau(3);
    t_upper(yy == 4) = tau(4);
    t_lower(yy == 5) = tau(4);
    t_upper(yy == 5) = inf;

    v_A_ext = v_A_ext_mat(:);
    z_A_ext = z_A_ext_mat(:);

    [z_B_post, v_B_post] = GaussianMomentsComputation_MJH_len(yy, z_A_ext, v_A_ext, wvar,Quantize_stepsize,t_lower,t_upper);

    z_B_post_mat = reshape(z_B_post,m,n);
    v_B_post_mat = reshape(v_B_post,m,n);

    v_B_ext_mat = v_B_post_mat.*v_A_ext_mat./(v_A_ext_mat-v_B_post_mat);

    v_B_ext_mat = var_max*(v_B_ext_mat<=0)+v_B_ext_mat.*(v_B_ext_mat>0);
    v_B_ext_mat = min(v_B_ext_mat,var_max);
    v_B_ext_mat = max(v_B_ext_mat,var_min);

    z_B_ext_mat = v_B_ext_mat.*(z_B_post_mat./v_B_post_mat-z_A_ext_mat./v_A_ext_mat);

    Cov_Z = nan(m,n);

    Y_tilde = P.*z_B_ext_mat;
    Y2sum = sum(abs(Y_tilde(:)).^2);
    scale2 = Y2sum / (mn);  % variance of Y
    % scale = sqrt(scale2);
    scale = sqrt(scale2)/10;

    [U, S, V] = svd(Y_tilde, 'econ');
    A = U(:,1:r0)*(S(1:r0,1:r0)).^(0.5);
    B = (S(1:r0,1:r0)).^(0.5)*V(:,1:r0)';
    B = B';

    Sigma_A = repmat( scale*eye(r0,r0), [1 1 m] );%一共m行，每一行的协方差矩阵
    Sigma_B = repmat( scale*eye(r0,r0), [1 1 n] );

    gammas = (m + n)./( diag(B'*B) + diag(sum(Sigma_B,3)) + diag(A'*A)+ diag(sum(Sigma_A,3)) );   

    beta = 1./v_B_ext_mat;

    Z_hat = A*B';

    [ Y_hat ] = Quantizey(Z_hat, tau);
    
    for iter_outer = 1:T_max-1
        Aw_inv = diag(gammas);   % 列向量化为对角阵
        betaY = beta.*z_B_ext_mat;

        for inner = 1:1
            %% A step
            for i=1:m %iterate over rows
                observed = find(P(i,:));
                Bi = B(observed,:);
                Bibeta = zeros(size(Bi));
                Bibeta = bsxfun(@times,Bi,sqrt(beta(i,observed)'));
                Sigma_Bbeta = bsxfun(@times,Sigma_B(:,:,observed),reshape(beta(i,observed),1,1,[]));
                Sigma_A(:,:,i) = (Bibeta'*Bibeta + sum(Sigma_Bbeta,3) + Aw_inv)^(-1);
                A(i,:) = betaY(i,observed)*Bi*Sigma_A(:,:,i);
            end

            %% B step
            for j=1:n %Iterate over cols
                observed = find(P(:,j));
                Aj = A(observed,:);
                Ajbeta = zeros(size(Aj));
                Ajbeta = bsxfun(@times,Aj,sqrt(beta(observed,j)));
                Sigma_Abeta = bsxfun(@times,Sigma_A(:,:,observed),reshape(beta(observed,j),1,1,[]));
                Sigma_B(:,:,j) = (Ajbeta'*Ajbeta + sum(Sigma_Abeta,3) + Aw_inv)^(-1);
                B(j,:) = betaY(observed,j)'*Aj*Sigma_B(:,:,j);
            end

            %% estimate gammas
            gammas = (m + n)./( diag(B'*B) + diag(sum(Sigma_B,3)) + diag(A'*A)+ diag(sum(Sigma_A,3)) );

        end
        %% update X
        Z_hat = A*B';

        Cov_Z = nan(m,n);
        [ra,ra,~] = size(Sigma_A);
        [rb,rb,~] = size(Sigma_B);
        Sigma_A_r = permute(Sigma_A,[2,1,3]);
        Sigma_A_t = reshape(Sigma_A_r,1,ra*ra,m);
        Sigma_A_t = squeeze(Sigma_A_t);
        Sigma_A_t = Sigma_A_t.';
        Sigma_B_t = reshape(Sigma_B,rb*rb,n);

        BsigmaA = pagemtimes(B,Sigma_A);
        BsigmaAB0 = bsxfun(@times,BsigmaA,conj(B));
        BsigmaAB_sum = sum(BsigmaAB0,2);
        BsigmaABre = reshape(BsigmaAB_sum,[n,m]);
        BsigmaAB = BsigmaABre.';

        AsigmaB = pagemtimes(A,Sigma_B);
        AsigmaBA0 = bsxfun(@times,AsigmaB,conj(A));
        AsigmaBA_sum = sum(AsigmaBA0,2);
        AsigmaBAre = reshape(AsigmaBA_sum,[m,n]);

        Cov_Z = BsigmaAB + AsigmaBAre +Sigma_A_t*Sigma_B_t;

        Cov_Z = (Cov_Z+conj(Cov_Z))/2;

        if(iter_outer>1)
            v_A_post_mat0 = P.*real(Cov_Z);
            wvar = (norm((z_B_ext_mat.*P-Z_hat.*P),'fro').^2 + sum(v_A_post_mat0(:)))/(sum(P(:)));
        end
        z_A_ext_mat_old = Z_hat;
        v_A_post_mat_old = real(Cov_Z);

        v_A_ext_mat = real(Cov_Z).*v_B_ext_mat./(v_B_ext_mat-real(Cov_Z));

        v_A_ext_mat = var_max*(v_A_ext_mat<=0)+v_A_ext_mat.*(v_A_ext_mat>0);
        v_A_ext_mat = max(v_A_ext_mat,var_min);
        v_A_ext_mat = min(v_A_ext_mat,var_max);

        z_A_ext_mat = v_A_ext_mat.*(Z_hat./Cov_Z-z_B_ext_mat./v_B_ext_mat);
        v_A_ext = v_A_ext_mat(:);
        z_A_ext = z_A_ext_mat(:);

        [z_B_post, v_B_post] = GaussianMomentsComputation_MJH_len(yy, z_A_ext, v_A_ext, wvar,Quantize_stepsize, t_lower,t_upper);

        z_B_post_mat = reshape(z_B_post,m,n);
        v_B_post_mat = reshape(v_B_post,m,n);

        v_B_ext_mat = v_B_post_mat.*v_A_ext_mat./(v_A_ext_mat-v_B_post_mat);

        v_B_ext_mat = var_max*(v_B_ext_mat<=0)+v_B_ext_mat.*(v_B_ext_mat>0);
        v_B_ext_mat = min(v_B_ext_mat,var_max);
        v_B_ext_mat = max(v_B_ext_mat,var_min);

        z_B_ext_mat = v_B_ext_mat.*(z_B_post_mat./v_B_post_mat-z_A_ext_mat./v_A_ext_mat);

        v_B_ext_mat_old = v_B_ext_mat;
        z_B_ext_mat_old = z_B_ext_mat;

        beta = 1./v_B_ext_mat;

        [ Y_hat ] = Quantizey(Z_hat, tau);
    end
    RMSE(mc) = sqrt(sum(sum((X_true.*index_test -Y_hat.*index_test).^2))/sum(sum(index_test)));
    NMAE(mc) = sum(sum(abs(X_true.*index_test -Y_hat.*index_test)))/(sum(sum(index_test))*(X_max-X_min));
    clc
end


save('movie_len_GR_SBL_4.mat', "RMSE", "NMAE")

