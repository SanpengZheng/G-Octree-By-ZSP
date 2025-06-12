clear
%% Experiment for Section 4.1
% N=100000, 200000, 400000, 800000, 1600000
% For each N, calculate the approximation at 100 random points.
%%
d=2;
m=3;
f=@(x) 0.75*exp(-((9*x(1)-2)^2+(9*x(2)-2)^2)/4)+0.75*exp(-(9*x(1)+1)^2/49-(9*x(2)+1)/10)...
    +0.5*exp(-((9*x(1)-7)^2+(9*x(2)-3)^2)/4)-0.2*exp(-(9*x(1)-4)^2-(9*x(2)-7)^2); % Franke test function
Box=[zeros(d,1),ones(d,1)];
TpOfkdtree=zeros(5,100);
TpOfGoctree=zeros(5,100);
TpOfPMLS=zeros(5,100);
TsOfMLS=zeros(5,100);
TsOfkdtree=zeros(5,100);
TsOfGoctree=zeros(5,100);
TsOfPMLS=zeros(5,100);
EOfMLS=zeros(5,100);
EOfkdtree=zeros(5,100);
EOfGoctree=zeros(5,100);
EOfPMLS=zeros(5,100);
for i=1:5
    %%
    N=2^(i-1)*100000;
    Xn=Halton(d,N); %Halton points
    h=1/(N^(1/d)); %the approximate filling distance
    delta=2*m*h; % formula (4.1)
    Fs=zeros(N,1);
    for j=1:N
        Fs(j)=f(Xn(:,j));
    end
    for j=1:100
        %% Preparation
        tic
        Mdl = KDTreeSearcher(Xn');
        TpOfkdtree(i,j)=toc; %tree-building time of kd-tree
        tic
        octree=MyOctree(Xn,Box,delta);
        TpOfGoctree(i,j)=toc; %tree-building time of G-Otree
        tic
        dh=2*delta/sqrt(d);
        PMLS=PiecewiseMLSByZsp(Xn,Fs,Box,ceil(1/dh)*ones(d,1),m,delta); %Preparation time of PMLS
        TpOfPMLS(i,j)=toc; 
        %% The single approximation time
        Tj=rand(d,1); % A random point
        fr=f(Tj);
        %% MLS without using acceleration methods
        tic
        [Alpha,Q] = PolyAlpha(d,m);
        Idx1=false(1,N);
        W=zeros(N,1);
        P=zeros(N,Q);
        for k=1:N
            W(k)=LW(Tj,Xn(:,k),delta);
            if W(k)>0
                Idx1(k)=true;
                P(k,:)=SSPnByAlpha(Xn(:,k),Tj,delta,Alpha);
            end
        end
        Idx1=find(Idx1>0);
        Wj=diag(sqrt(W(Idx1)));
        P=P(Idx1,:);
        C1=SbySVD(Wj*P,Wj*Fs(Idx1));
        fp=SSPnByAlpha(Tj,Tj,delta,Alpha)*C1;
        TsOfMLS(i,j)=toc;
        EOfMLS(i,j)=abs(fr-fp);
        %% MLS accelerated by kd-tree
        tic
        [Alpha,Q] = PolyAlpha(d,m);
        Idx2 = rangesearch(Mdl,Tj',delta);
        Idx2=Idx2{1};
        Nj=length(Idx2);
        W=zeros(Nj,1);
        P=zeros(Nj,Q);
        for k=1:Nj
            W(k)=LW(Tj,Xn(:,Idx2(k)),delta);
            P(k,:)=SSPnByAlpha(Xn(:,Idx2(k)),Tj,delta,Alpha);
        end
        Wj=diag(sqrt(W));
        C2=SbySVD(Wj*P,Wj*Fs(Idx2));
        fp=SSPnByAlpha(Tj,Tj,delta,Alpha)*C2;
        TsOfkdtree(i,j)=toc;
        EOfkdtree(i,j)=abs(fr-fp);
        %% MLS accelerated by G-Octree
        tic
        [Alpha,Q] = PolyAlpha(d,m);%使用的多项式基底指标
        Idx3=octree.RangeSearch(Tj,delta);
        Nj=length(Idx3);
        W=zeros(Nj,1);
        P=zeros(Nj,Q);
        for k=1:Nj
            W(k)=LW(Tj,Xn(:,Idx3(k)),delta);
            P(k,:)=SSPnByAlpha(Xn(:,Idx3(k)),Tj,delta,Alpha);
        end
        Wj=diag(sqrt(W));
        C3=SbySVD(Wj*P,Wj*Fs(Idx3));
        fp=SSPnByAlpha(Tj,Tj,delta,Alpha)*C3;
        TsOfGoctree(i,j)=toc;
        EOfGoctree(i,j)=abs(fr-fp);
        %% PMLS
        tic
        fp=PMLS.Aprox(Tj);
        TsOfPMLS(i,j)=toc;
        EOfPMLS(i,j)=abs(fr-fp);
    end
end
%% Results of Table 1 in Sec. 4.1
MeanOfEOfMLS=mean(EOfMLS,2);
StdOfEOfMLS=std(EOfMLS,0,2);
MeanOfEOfkdtree=mean(EOfkdtree,2);
StdOfEOfkdtree=std(EOfkdtree,0,2);
MeanOfEOfGoctree=mean(EOfGoctree,2);
StdOfEOfGoctree=std(EOfGoctree,0,2);
MeanOfEOfPMLS=mean(EOfPMLS,2);
StdOfEOfPMLS=std(EOfPMLS,0,2);
%% Results of Table 2 in Sec. 4.1
MeanOfTpfkdtree=mean(TpOfkdtree,2);
StdOfTpOfkdtree=std(TpOfkdtree,0,2);
MeanOfTpOfGoctree=mean(TpOfGoctree,2);
StdOfTpOfGoctree=std(TpOfGoctree,0,2);
MeanOfTpOfPMLS=mean(TpOfPMLS,2);
StdOfTpOfPMLS=std(TpOfPMLS,0,2);
%% Results of Table 3 in Sec. 4.1
MeanOfTsOfMLS=mean(TsOfMLS,2);
StdOfTsOfMLS=std(TsOfMLS,0,2);
MeanOfTsOfkdtree=mean(TsOfkdtree,2);
StdOfTsOfkdtree=std(TsOfkdtree,0,2);
MeanOfTsOfGoctree=mean(TsOfGoctree,2);
StdOfTsOfGoctree=std(TsOfGoctree,0,2);
MeanOfTsOfPMLS=mean(TsOfPMLS,2);
StdOfTsOfPMLS=std(TsOfPMLS,0,2);
%% Results of Table 3 in Sec. 4.1 
% The total time of each method which contain one preparation and 100 approximations
TotalTimeOfMLS=sum(TsOfMLS,2);
TotalTimeOfkdtree=MeanOfTpfkdtree+sum(TsOfkdtree,2);
TotalTimeOfGoctree=MeanOfTpOfGoctree+sum(TsOfGoctree,2);
TotalTimeOfPMLS=MeanOfTpOfPMLS+sum(TsOfPMLS,2);
save('ResultsOfExperimentForSec4.1.mat')
%% Auxiliary functions
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
function Xn = Halton(d,N)
%Halton(d,N) Generate N d-dimensional halton points and store them as a column vector matrix
p=haltonset(d,'Skip',1e3,'Leap',1e2);
D=net(p,N);
Xn=D';
end