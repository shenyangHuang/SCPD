function ldos = compute_ldos(normalized_adj, v, Nbins, iterations)
    if nargin<4
        iterations = 100;
    end
    
    if nargin<3
        Nbins = 50;
    end
    
    if norm(v) == 0
        ldos = zeros(1,Nbins);
    else
        v = v/norm(v);
        [theta,wts] = moments_lanczos(normalized_adj,v,iterations,1e-6);
        c = moments_quad2cheb(theta, wts, 20);
        cf = filter_jackson(c);
        ldos = plot_cheb_ldos(Nbins+1,cf);
        %ldos = max(ldos,0);
    end
    
end

