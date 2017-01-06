[V, m, n, nnz] = mmread('~/0/so_x_p_proj5_1.mmc');

[W,D,H] = svds(V,50);

[W,D,H] = svds(V,200);

mmwrite('so_x_p_proj5_1-w-200-svd.mma', W*sqrt(D));

mmwrite('so_x_p_proj5_1-h-200-svd.mma', sqrt(D)*H');
