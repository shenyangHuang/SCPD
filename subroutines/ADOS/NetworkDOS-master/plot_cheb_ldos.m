% yy = plot_cheb_ldos(c,xx,ab)
%
% Given a set of first-kind Chebyshev moments, compute the associated
% density.   If no output argument is assigned, make a plot.
%
% Inputs:
%   c:  Array of Chebyshev moments (on [-1,1])
%   xx: evaluation points (defaults to mesh of 1001 pts)
%   ab: mapping parameters (defaults to identity)
%
% Output:
%   yy: Density evaluated at xx mesh (size nnodes-by-nmesh)
%   idx: Index for spectral re-ordering

function [yy,idx] = plot_cheb_ldos(varargin)


% Parse arguments
[c,xx,xx0,ab] = plot_cheb_argparse(varargin{1}, varargin{2:end});
% npts=1001;
%     ab = [1, 0];
%     xx0 = linspace(-1+1e-8,1-1e-8,npts);
%   
%     xx = xx0;
%     c=varargin{1};
    
% Run the recurrence to compute CDF
[nmoment, nnodes] = size(c);
txx = acos(xx);
yy = c(1,:)'*(txx-pi)/2;
for np = 2:nmoment
    n = np-1;
    yy = yy + c(np,:)' * sin(n*txx)/n;
end
yy = -2/pi * yy;

% Difference the CDF to compute histogram
yy = yy(:,2:end)-yy(:,1:end-1);


%Compute sorted histogram
if nargout ~= 1
    [U,S,V] = svd(yy);
    [~,idx] = sort(U(:,1));
    idx
    
end

% Plot if appropriate
if nargout < 1
    yr = [1, nnodes];
    xr = [xx0(1)+xx0(2), xx0(end-1)+xx0(end)]/2; %x-range
    figure('outerposition',[0 0 1050 900]);
    bot = min(min(yy)); top = max(max(yy));
    
    if(nnodes==1)
         
       bar([xx0(1:length(xx0)-1)+xx0(2:end)]/2,yy)
       xlabel('\lambda')
        ylabel('hist')
     set(gca,'xtick',linspace(-1,1,11),'FontSize',20,'FontWeight','bold');
     box on;
   
    else
        size(yy)
        pause
   imagesc(xr, yr, yy(idx,:));
   colormap('jet');
   caxis manual
   caxis([bot top])
   colorbar;
    xlabel('\lambda')
    ylabel('Node Index')
    set(gca,'xtick',linspace(-1,1,11),'FontSize',20,'FontWeight','bold');
    box on;
    end
end