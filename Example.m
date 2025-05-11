%%
% An example of using G-Octree for neighborhood search and moving least squares (MLS) approximation
%%
% Preparation
clear
d=2;
m=3;
N=160*10000;
Xn=rand(d,N);   % Sampling point set
h=1/(N^(1/d)); % The estimation of fill distance
delta0=2*m*h;
f=@(x) 0.75*exp(-((9*x(1)-2)^2+(9*x(2)-2)^2)/4)+0.75*exp(-(9*x(1)+1)^2/49-(9*x(2)+1)/10)...
    +0.5*exp(-((9*x(1)-7)^2+(9*x(2)-3)^2)/4)-0.2*exp(-(9*x(1)-4)^2-(9*x(2)-7)^2); % Franke test function
Fs=zeros(N,1); % Sampling values
for i=1:N
    Fs(i)=f(Xn(:,i));
end
%%
% Main
% Building G-Octree
C0=[zeros(d,1),ones(d,1)];
GOctree=MyOctree(Xn,C0,delta0);
%
% Neighborhood search
x=rand(d,1);
%delta=delta0*(0.5+rand(1,1));   % The radius of search, delta, is a random number between 0.5*delta0 and 1.5*delta0
delta=8*delta0;
Ix=GOctree.RangeSearch(x,delta); % The index set of pionts in the intersection intersection of Xn and B(x,delta)
%%
% MLS approximation at x
[Alpha,Q] = PolyAlpha(d,m);
Nx=length(Ix);
W=zeros(Nx,1);
P=zeros(Nx,Q);
for k=1:Nx
    W(k)=LW(x,Xn(:,Ix(k)),delta);  % Weight function in MLS
    P(k,:)=PnByAlpha(Xn(:,Ix(k)),Alpha); % Measurement matrix of polynomial basis on Xn
end
Wx=diag(sqrt(W));
c=SbySVD(Wx*P,Wx*Fs(Ix));
fp=PnByAlpha(x,Alpha)*c;
Error=abs(fp-f(x)); %Approximation error at x

%%
% Auxiliary functions
function [A,Q] = PolyAlpha(d,m)
%PolyAlpha Return a multi-index set of d-element polynomials with a total degree not higher than m based on the input
Q=nchoosek(d+m,d);
A=zeros(Q,d);
a=zeros(1,d);
i=2;
while i<=Q
    a=AddOne(a,m);
    if sum(a)<=m
        A(i,:)=a;
        i=i+1;
    end
end
    function b=AddOne(a,m)
        N=length(a);
        H=a(1);
        if H>m
            error('e1')
        else
            Ha=H+1;
            if Ha<m+1
                b=a;
                b(1)=Ha;
            else
                if N<=1
                    error('e2')
                else
                    b=zeros(1,N);
                    b(2:N)=AddOne(a(2:N),m);
                end
            end
        end
    end
end
function p=PnByAlpha(X,Alpha)
%PnByAlpha The values of the polynomial basis at X under the multi-index set Alpha
[Q,d]=size(Alpha);
if length(X)~=d
    error('e3')
else
    p=zeros(1,Q);
    x=X(:)';
    for i=1:Q
        p(i)=prod(x.^Alpha(i,:));
    end
end
end
function w =LW(x,y,delta)
%LW The weight funcion in MLS
%Wendland's CSRBF
n=length(x);
r=zeros(n,1);
for i=1:n
    r(i)=x(i)-y(i);
end
R=sqrt(r'*r);
R=R/delta;
if R<1
    w=((1-R)^6)*(35*R^2+18*R+3);
else
    w=0;
end
end
function c = SbySVD(A,b)
%SbySVD Solve c=A\b using singular value decomposition
[U,S,V] = svd(A); % A=U*S*V';
[l,h]=size(S);
iS=zeros(l,h);
for i=1:min([l,h])
    if S(i,i)>0
        iS(i,i)=1/S(i,i);
    end
end
c=(V*iS'*U')*b;
end