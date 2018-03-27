% Program 13.3 Nelder-Mead Search
% Input: function f, best guess xbar (column vector),
% initial search radius rad and number of steps k
% Output: matrix x whose columns are vertices of simplex,
% function values y of those vertices
function [x,y]=neldermead(f,xbar,rad,k)
n=length(xbar);
x(:,1)=xbar; % each column of x is a simplex vertex
x(:,2:n+1)=xbar*ones(1,n)+rad*eye(n,n);
for j=1:n+1
y(j)=f(x(:,j)); % evaluate obj function f at each vertex
end
[y,r]=sort(y); % sort the function values in ascending order
x=x(:,r); % and rank the vertices the same way
for i=1:k
xbar=mean(x(:,1:n)’)’; % xbar is the centroid of the face
xh=x(:,n+1); % omitting the worst vertex xh
xr = 2*xbar - xh; yr = f(xr);
if yr < y(n)
if yr < y(1) % try expansion xe
xe = 3*xbar - 2*xh; ye = f(xe);
if ye < yr % accept expansion
x(:,n+1) = xe; y(n+1) = f(xe);
else % accept reflection
x(:,n+1) = xr; y(n+1) = f(xr);
end
else % xr is middle of pack, accept reflection
x(:,n+1) = xr; y(n+1) = f(xr);
end
else % xr is still the worst vertex, contract
if yr < y(n+1) % try outside contraction xoc
xoc = 1.5*xbar - 0.5*xh; yoc = f(xoc);
if yoc < yr % accept outside contraction
x(:,n+1) = xoc; y(n+1) = f(xoc);
else % shrink simplex toward best point
for j=2:n+1
x(:,j) = 0.5*x(:,1)+0.5*x(:,j); y(j) = f(x(:,j));
end
end
else % xr is even worse than the previous worst
xic = 0.5*xbar+0.5*xh; yic = f(xic);
if yic < y(n+1) % accept inside contraction
x(:,n+1) = xic; y(n+1) = f(xic);
else % shrink simplex toward best point
for j=2:n+1
x(:,j) = 0.5*x(:,1)+0.5*x(:,j); y(j) = f(x(:,j));
end
end
end
end
[y,r] = sort(y); % resort the obj function values
x=x(:,r); % and rank the vertices the same way
end
The code implements the flowchart in Figure 13.5(b). The number of iteration steps is
required as an input. Computer Problem 8 asks the reader to rewrite the codewith a stopping
criterion based on a user-given error tolerance. A common stopping criterion is to require
574 | CHAPTER 13 Optimization
0
1
0
1
y
15
z
x
0
5
10
1
1
Figure 13.6 Surface plot of two-dimensional function. Graph of
z = 5x4 +4x2y − xy3 +4y4 − x. Minimum is found by the Nelder-Mead method to
occur at ≈ ( 0.4923, − 0.3643).
both that the simplex has reduced in size to within a small distance tolerance and that the
maximum spread of the function values at the vertices is within a small tolerance. Matlab
implements the Nelder–Mead Method in its fminsearch command.
! EXAMPLE 13.3 Locate the minimum of the function f (x,y) = 5x4 + 4x2y − xy3 + 4y4 − x, using the