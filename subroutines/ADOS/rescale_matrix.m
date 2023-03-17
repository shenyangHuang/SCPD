% [N] = rescale_matrix(L, mode)
%
% Output:
%   N: a normalized Laplacian matrix or stochastic matrix (in sparse form)
%
function [N] = rescale_matrix(L, low, high)
    I = sparse(eye(size(L,1)));
    ab = [(high-low)/2;(high+low)/2];
    N = (L-ab(2)*I)/ab(1);
end
