% [L] = normalize_laplacian(L, A)
% Normalize a Laplacian matrix
% Input:
%   L: unnormalized Laplacian matrix
%   degrees: degree sequence
function [L] = normalize_laplacian(L, degrees)
    [i,j,lij] = find(L);
    %lij = lij./sqrt(degrees(i).*degrees(j));
    lij = lij./(degrees(i).*degrees(j));
end
