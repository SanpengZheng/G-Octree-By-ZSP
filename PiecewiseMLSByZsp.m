classdef PiecewiseMLSByZsp
    % PiecewiseMLSByZsp The method in "W. Li, G. Song, G. Yao, Piece-wise moving least
    % square approximation, Appl. Numer. Math. 115 (2017) 68–81,
    % doi:10.1016/j.apnum.2017.01.001." implemented by Sanpeng Zheng
    properties(SetAccess=private)
        DN
        box
        dB
        dBm
        Cp
        Alpha
        Tn
        r
    end
    methods
        function obj = PiecewiseMLSByZsp(Xn,Fn,box,DN,m,r)
            obj.dB=(box(:,2)-box(:,1));
            obj.dBm=obj.dB./DN;
            obj.box=box;
            obj.DN=DN;
            obj.r=r;
            %%  The calculation of coefficients is accelerated by kd-tree
            Mdl = KDTreeSearcher(Xn');
            %%
            % Calculate the anchor points and the corresponding MLS approximation coefficients
            Nmax=prod(DN);
            [d,~]=size(Xn);
            [obj.Alpha,Q] = PolyAlpha(d,m);
            obj.Cp=zeros(Q,Nmax);
            obj.Tn=zeros(d,Nmax);
            Di=ones(d,1);
            Di(1)=0;
            for i=1:Nmax
                Di=AddOne(Di,DN);
                Ti=obj.box(:,1)+(Di-1/2).*obj.dBm;
                obj.Tn(:,i)=Ti;
                Idxi = rangesearch(Mdl,Ti',r); 
                Idxi=Idxi{1};
                Ni=length(Idxi);
                W=zeros(Ni,1);
                P=zeros(Ni,Q);
                for k=1:Ni
                    W(k)=LW(Ti,Xn(:,Idxi(k)),r);
                    P(k,:)=SSPnByAlpha(Xn(:,Idxi(k)),Ti,r,obj.Alpha);
                end
                Wi=diag(sqrt(W));
                obj.Cp(:,i)=SbySVD(Wi*P,Wi*Fn(Idxi));
            end
        end
        function Pf = Aprox(obj,x)
            %Locate the number of the subspace in where x is
            d=length(x);
            Dk=ceil((x-obj.box(:,1))./obj.dBm);Dk(Dk<=0)=1;
            k=Dk(1);
            ns=1;
            for i=2:d
                ns=ns*obj.DN(i-1);
                k=k+ns*(Dk(i)-1);
            end
            Pf=SSPnByAlpha(x,obj.Tn(:,k),obj.r,obj.Alpha)*obj.Cp(:,k);
        end
    end
end
%% Auxiliary functions
function D=AddOne(D,DN)
%AddOne The vector OD takes DM as the upper limit of each component, adding one from the last digit
if D(1)<DN(1)
    D(1)=D(1)+1;
else
    if D(1)>DN(1)
        error('Reache the upper limit. Can not add one')
    else
        if isscalar(D)
            error('Reache the upper limit. Can not add one')
        else
            D(1)=1;
            D(2:end)=AddOne(D(2:end),DN(2:end));
        end
    end
end
end
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
            error('Reache the upper limit. Can not add one')
        else
            Ha=H+1;
            if Ha<m+1
                b=a;
                b(1)=Ha;
            else
                if N<=1
                    error('Reache the upper limit. Can not add one')
                else
                    b=zeros(1,N);
                    b(2:N)=AddOne(a(2:N),m);
                end
            end
        end
    end
end

function p=SSPnByAlpha(X,C,delta,Alpha)
%SSPnByAlpha The values of the shifted and scaled polynomial basis at X under the multi-index set Alpha
[Q,d]=size(Alpha);
if length(X)~=d
    error('e3')
else
    SSX=(X-C)./delta;
    p=zeros(1,Q);
    x=SSX(:)';
    for i=1:Q
        p(i)=prod(x.^Alpha(i,:));
    end
end
end
function w =LW(x,y,delta)
%LW The weight funcion in MLS
% Wendland's CSRBF
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