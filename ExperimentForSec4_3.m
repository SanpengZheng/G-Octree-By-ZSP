clear
%% Experiment for Section 4.3
% N=225*10000, d=2 but delta=lambda*delta0 where lambda in (0,4]
% For each delta, calculate the approximation at 100 random points.
%%
d=2;
m=3;
f=@(x) 0.75*exp(-((9*x(1)-2)^2+(9*x(2)-2)^2)/4)+0.75*exp(-(9*x(1)+1)^2/49-(9*x(2)+1)/10)...
    +0.5*exp(-((9*x(1)-7)^2+(9*x(2)-3)^2)/4)-0.2*exp(-(9*x(1)-4)^2-(9*x(2)-7)^2); % Franke test function
%% Generate the non-uniform sampling point set
n=10000;
X=Halton(d,64*n);%The following auxiliary function Halton(d,N) is used. 
% Therefore, if you want to run this part of the codes independently, you need to add them to a new script.
X=0.25*X;
N=225*n;
Xn=zeros(d,N);
Xn(:,1:n)=X(:,1:n);
Xn(:,n+1:3*n)=X(:,1:2*n)+[0.25;0];
Xn(:,3*n+1:5*n)=X(:,2*n+1:4*n)+[0;0.25];
Xn(:,5*n+1:9*n)=X(:,1:4*n)+[0.5;0];
Xn(:,9*n+1:13*n)=X(:,4*n+1:8*n)+[0.25;0.25];
Xn(:,13*n+1:17*n)=X(:,8*n+1:12*n)+[0;0.5];
Xn(:,17*n+1:25*n)=X(:,1:8*n)+[0.75;0];
Xn(:,25*n+1:33*n)=X(:,8*n+1:16*n)+[0.5;0.25];
Xn(:,33*n+1:41*n)=X(:,16*n+1:24*n)+[0.25;0.5];
Xn(:,41*n+1:49*n)=X(:,24*n+1:32*n)+[0;0.75];
Xn(:,49*n+1:65*n)=X(:,1:16*n)+[0.75;0.25];
Xn(:,65*n+1:81*n)=X(:,16*n+1:32*n)+[0.5;0.5];
Xn(:,81*n+1:97*n)=X(:,32*n+1:48*n)+[0.25;0.75];
Xn(:,97*n+1:129*n)=X(:,1:32*n)+[0.75;0.5];
Xn(:,129*n+1:161*n)=X(:,32*n+1:64*n)+[0.5;0.75];
Xn(:,161*n+1:225*n)=X(:,1:64*n)+[0.75;0.75];
%%
Fs=zeros(N,1);
for i=1:N
    Fs(i)=f(Xn(:,i));
end
hF=1/(N^(1/d));   % the approximate filling distance
delta0=2*m*hF;    % formula (4.1)
Box=[zeros(d,1),ones(d,1)];
Noflambda=400;
TpOfkdtree=zeros(1,100);
TpOfGoctree=zeros(1,100);
TsOfMLS=zeros(Noflambda,100);
TsOfkdtree=zeros(Noflambda,100);
TsOfGoctree=zeros(Noflambda,100);
EOfMLS=zeros(Noflambda,100);
EOfkdtree=zeros(Noflambda,100);
EOfGoctree=zeros(Noflambda,100);
CondOfNormalMatrix=zeros(Noflambda,100);
for j=1:100
    %% Preparation
    tic
    Mdl = KDTreeSearcher(Xn');
    TpOfkdtree(j)=toc; %tree-building time of kd-tree
    tic
    octree=MyOctree(Xn,Box,delta);
    TpOfGoctree(j)=toc; %tree-building time of G-Otree
    %% The single approximation time
    Tj=rand(d,1); % A random point
    fr=f(Tj);
    for i=1:Noflambda
        delta=(0.01*i)*delta0; % delta=lambda*delta0
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
        CondOfNormalMatrix(i,j)=cond(P'*Wj*P,2);
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
        [Alpha,Q] = PolyAlpha(d,m);
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
    end
end
%% Results of Fig.6 in Sec. 4.3
% For Fig.6(a)
n=100;
N=225*n;
d=2;
X=Halton(d,64*n); %The following auxiliary function Halton(d,N) is used. 
% Therefore, if you want to run this part of the codes independently, you need to add them to a new script.
X=0.25*X;
hF=1/((N)^(1/d));
Xn=zeros(d,N);
Xn(:,1:n)=X(:,1:n);
Xn(:,n+1:3*n)=X(:,1:2*n)+[0.25;0];
Xn(:,3*n+1:5*n)=X(:,2*n+1:4*n)+[0;0.25];
Xn(:,5*n+1:9*n)=X(:,1:4*n)+[0.5;0];
Xn(:,9*n+1:13*n)=X(:,4*n+1:8*n)+[0.25;0.25];
Xn(:,13*n+1:17*n)=X(:,8*n+1:12*n)+[0;0.5];
Xn(:,17*n+1:25*n)=X(:,1:8*n)+[0.75;0];
Xn(:,25*n+1:33*n)=X(:,8*n+1:16*n)+[0.5;0.25];
Xn(:,33*n+1:41*n)=X(:,16*n+1:24*n)+[0.25;0.5];
Xn(:,41*n+1:49*n)=X(:,24*n+1:32*n)+[0;0.75];
Xn(:,49*n+1:65*n)=X(:,1:16*n)+[0.75;0.25];
Xn(:,65*n+1:81*n)=X(:,16*n+1:32*n)+[0.5;0.5];
Xn(:,81*n+1:97*n)=X(:,32*n+1:48*n)+[0.25;0.75];
Xn(:,97*n+1:129*n)=X(:,1:32*n)+[0.75;0.5];
Xn(:,129*n+1:161*n)=X(:,32*n+1:64*n)+[0.5;0.75];
Xn(:,161*n+1:225*n)=X(:,1:64*n)+[0.75;0.75];
figure(1)
plot(Xn(1,:),Xn(2,:),'.','MarkerSize',1)
% For Fig.6(b)
[X,Y]=meshgrid(0:0.25:1,0:0.25:1);
figure(2)
plot(X,Y,'k','LineWidth',2.5)
hold on 
plot(X',Y','k','LineWidth',2.5)
P0=[0.09,0.16];
text('Position',P0,'String','n','FontSize',42)
text('Position',P0+[0.2,0],'String','2n','FontSize',42)
text('Position',P0+[-0.05,0.25],'String','2n','FontSize',42)
text('Position',P0+[0.45,0],'String','4n','FontSize',42)
text('Position',P0+[0.2,0.25],'String','4n','FontSize',42)
text('Position',P0+[-0.05,0.5],'String','4n','FontSize',42)
text('Position',P0+[0.7,0],'String','8n','FontSize',42)
text('Position',P0+[0.45,0.25],'String','8n','FontSize',42)
text('Position',P0+[0.2,0.5],'String','8n','FontSize',42)
text('Position',P0+[-0.05,0.75],'String','8n','FontSize',42)
text('Position',P0+[0.675,0.25],'String','16n','FontSize',42)
text('Position',P0+[0.425,0.5],'String','16n','FontSize',42)
text('Position',P0+[0.175,0.75],'String','16n','FontSize',42)
text('Position',P0+[0.675,0.5],'String','32n','FontSize',42)
text('Position',P0+[0.425,0.75],'String','32n','FontSize',42)
text('Position',P0+[0.675,0.75],'String','64n','FontSize',42)
%% Results of Fig.7 in Sec. 4.3
k0=0;
for j=1:100
 I=find(CondOfNormalMatrix(:,j)==inf);
 if ~isempty(I)
   mI=max(I);
   if mI>k0
       k0=mI; 
   end
 end
end
[Emin,km]=min(sum(EOfMLS,2)/100);
lambda=0.01:0.01:4;
figure(3);
semilogy(lambda(k0:end),sum(EOfMLS(k0:end,:),2)/100,'LineWidth',1)
hold on
plot(lambda(km),Emin,'*r')
text(lambda(km), Emin+0.00000001, ['(', num2str(lambda(km)), ',', num2str(Emin), ')'],'FontSize',14);
xlabel('{\lambda}','Fontname', 'Times New Roman','FontSize',14)
ylabel('the mean of approximation error','Fontname', 'Times New Roman','FontSize',14)
hold off
%% Results of Fig.8 in Sec. 4.3
figure(4);
semilogy(lambda(k0:end),sum(CondOfNormalMatrix(k0:end,:),2)/100,'LineWidth',1)
xlabel('{\lambda}','Fontname', 'Times New Roman','FontSize',14)
ylabel('the mean of the condition number','Fontname', 'Times New Roman','FontSize',14)
hold on
Ckm=sum(CondOfNormalMatrix(km,:))/100;
plot(lambda(km),Ckm,'*r')
text(lambda(km), Ckm+100000, ['(', num2str(lambda(km)), ',', num2str(Ckm), ')'],'FontSize',14);
hold off
%% Results of Fig.9 in Sec. 4.3
figure(5);
semilogy(lambda(k0:end),sum(TsOfMLS(k0:end,:),2)/100,'LineWidth',1.5)
hold on
plot(lambda(k0:end),sum(TsOfkdtree(k0:end,:),2)/100,'-g','LineWidth',1.5)
plot(lambda(k0:end),sum(TsOfGoctree(k0:end,:),2)/100,'-r','LineWidth',1.5)
legend('MLS','MLS by kd-tree','MLS by G-Octree')
hold off
xlabel('{\lambda}','Fontname', 'Times New Roman','FontSize',14)
ylabel('the mean of single approximation time','Fontname', 'Times New Roman','FontSize',14)
hold off
%% Results of Table 8 in Sec. 4.3
MeanOfTpfkdtree=mean(TpOfkdtree,2);
MeanOfTpOfGoctree=mean(TpOfGoctree,2);
% The total time of each method which contain one preparation and 40000 approximations
TotalTimeOfMLS=sum(TsOfMLS,'all');
TotalTimeOfkdtree=MeanOfTpfkdtree+sum(TsOfkdtree,'all');
TotalTimeOfGoctree=MeanOfTpOfGoctree+sum(TsOfGoctree,'all');
save('ResultsOfExperimentForSec4.3.mat')
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