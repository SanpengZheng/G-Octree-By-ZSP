classdef MyOctree
    %MYOCTREE  A class to accelerate the neighborhood search in MLS
    properties
        points    % Store all points as column vectors
        Mbox      % The upper and lower limits of each component are stored in the form of D*2
        D         % Dimension
        Maps      % A tensor which stores the index of the index vector set
        idxs      % The index vector set
        MD        % Deepest level, the root node is defined as level 1
        R         %TThe shortest side lengths of each layer of the hyper-cuboid
        Nnum      %Number of neighbors for a inside node
        dN        %The degree of offset of the neighbor for the inside node is a matrix of D* (3^D)
    end
    methods
        function obj = MyOctree(X,box,delta0)
                       %Constructor build the G-Octree in paper
            %% Initialization
            obj.points=X;
            obj.Mbox=box;
            dB=(box(:,2)-box(:,1));
            R=zeros(16,1);
            R(1)=min(dB);
            MD=1;
            while true
            R(MD+1)=R(MD)/2;
                if R(MD+1)>delta0
                    MD=MD+1;
                else
                    break;
                end
            end
            obj.MD=MD;
            obj.R=R(1:MD);
            [obj.D,N]=size(X);
            Cn=2^obj.D;                                                     %The number of children
            MN=Cn^(MD-1);                                                   %Number of leaf nodes
            Dnmax=2^(MD-1)*ones(obj.D,1);
            obj.Maps=Dtensor(Dnmax);                                        %The pre-stored number 
            obj.idxs=cell(MN,1);
            Di=ones(obj.D,1);                                               %Level 1
            obj.idxs{obj.Maps.GetData(Di)}=1:1:N;                           %The indexs stored in root
            Dnold=ones(obj.D,1);
            Ddold=Dnmax(1);
            for i=2:obj.MD                                                  %Level by Level, store the index of indexs into Maps
                Dnnew=Dnold*2;
                Ddnew=Ddold/2;                                              %The edge vector in next level 
                Nold=Cn^(i-2);
                dB=dB./2;
                Dj=ones(obj.D,1);
                Dj(end)=0;
                for j=1:Nold                                        
                Dj=AddOne(Dj,Dnold);  
                Mb=box(:,1)+(2*Dj-1).*dB; 
                Djm=(Dj-1)*Ddold+1; 
                obj= Divide(obj,Djm,Ddnew,Mb,1);
                end
                Dnold=Dnnew;
                Ddold=Ddnew;
            end
            %% Auxiliary quantity
            obj.Nnum=3^obj.D;
            DM=ones(obj.D,1);
            Dm=-DM;
            obj.dN=zeros(obj.D,obj.Nnum);
            obj.dN(:,1)=Dm;
            for i=2:obj.Nnum
                obj.dN(:,i)=AddOnem2M(obj.dN(:,i-1),Dm,DM);
            end
        end
        function [Dk,dBl]= SearchBox(obj,point,l)
            %SearchBox Search for the node that point belongs to in the tree in level l
            %   point is the obj.D-dimensional column vector
            %   Dk represents the coordinates of the leaf node in the obj.D-dimensional tensor and is the obj.D-dimensional column vector
            %   Db is the edge vector of a leaf node
            dBl=(obj.Mbox(:,2)-obj.Mbox(:,1))./(2^(l-1));
            Dk=ceil((point-obj.Mbox(:,1))./dBl);
            Dk(Dk<=0)=1;
        end
        function Idx= RangeSearch(obj,point,r)
            %RangeSearch Search for all idxs of the data point in B(obj,point ,r)
            %   point is the obj.D-dimensional column vector
            %   Idx is a row vector
           [~,N]=size(obj.points);
           IB=false(1,N);
            key0=0;
            for lr=1:obj.MD-1
                if obj.R(lr+1) < r
                    key0=1;
                    break
                end
            end
            if key0                                                        % whihc means lr < obj.MD
                dl=obj.MD-lr;                                              % The difference from the deepest level
                Cn=2^dl;
                Ctn=Cn^obj.D;                                              % The required number of leaf nodes for an inter node
                [Dk,dBl]= SearchBox(obj,point,lr);
                Bl=obj.Mbox(:,1)+(Dk-1).*dBl;
                Dl=point-Bl;
                Dr=dBl-Dl;
                NMax=2^(lr-1);
                r2=r*r;
                for i=1:obj.Nnum
                    key=1;
                    dr=0;
                    for j=1:obj.D
                        if obj.dN(j,i)<0
                            if Dl(j)>r || Dk(j)<2
                                key=0;
                                break;
                            end
                            dr=dr+Dl(j)^2;
                        else
                            if obj.dN(j,i)>0
                                if Dr(j)>r || Dk(j)>NMax-1
                                    key=0;
                                    break;
                                end
                                dr=dr+Dr(j)^2;
                            end
                        end
                    end
                    if key
                        if dr<=r2                                           % which means the hyper-cuboid Dk+obj.dN(:,i) in level lr intersect with B(x,r)
                            Dm=(Dk+obj.dN(:,i)-1)*Cn+1;
                            DM=(Dk+obj.dN(:,i))*Cn;
                            Dj=Dm;
                            In=obj.idxs{obj.Maps.GetData(Dj)};
                            IB(In(vecnorm(obj.points(:,In)-point)<=r))=true;
                            for j=2:Ctn
                             Dj=AddOnem2M(Dj,Dm,DM);
                             In=obj.idxs{obj.Maps.GetData(Dj)};
                             IB(In(vecnorm(obj.points(:,In)-point)<=r))=true;
                            end
                        end
                    end
                end
            else                                                            % whihc means lr = obj.MD
                Dk= SearchBox(obj,point,obj.MD);
                dBm=(obj.Mbox(:,2)-obj.Mbox(:,1))./(2^(obj.MD-1));
                Bl=obj.Mbox(:,1)+(Dk-1).*dBm;
                Dl=point-Bl;
                Dr=dBm-Dl;
                NMax=2^(obj.MD-1);
                r2=r*r;
                for i=1:obj.Nnum
                    key=1;
                    dr=0;
                    for j=1:obj.D
                        if obj.dN(j,i)<0
                            if Dl(j)>r || Dk(j)<2
                                key=0;
                                break;
                            end
                            dr=dr+Dl(j)^2;
                        else
                            if obj.dN(j,i)>0
                                if Dr(j)>r || Dk(j)>NMax-1
                                    key=0;
                                    break;
                                end
                                dr=dr+Dr(j)^2;
                            end
                        end
                    end
                    if key
                        if dr<=r2
                            In=obj.idxs{obj.Maps.GetData(Dk+obj.dN(:,i))};
                            IB(In(vecnorm(obj.points(:,In)-point)<=r))=true;
                        end
                    end
                end
            end
            Idx=find(IB>0);
        end
    end

    methods(Hidden=true,Access=private)
        function obj= Divide(obj,Djm,Ddnew,Mb,k)
            %Divide The nodes corresponding to Djm at layer i-1 are divided with Mb as the center line
            % One dimension at a time, currently divided into 2 in k bits, recursive call
            bk=obj.points(k,obj.idxs{obj.Maps.GetData(Djm)})>Mb(k);
            D2=Djm;
            D2(k)=D2(k)+Ddnew;
            %%
            obj.idxs{obj.Maps.GetData(D2)}=obj.idxs{obj.Maps.GetData(Djm)}(bk);
            obj.idxs{obj.Maps.GetData(Djm)}(bk)=[];
            if k<obj.D
                obj= Divide(obj,Djm,Ddnew,Mb,k+1);
                obj= Divide(obj,D2,Ddnew,Mb,k+1);
            end
        end
    end
end

function ND=AddOne(OD,DM)
%AddOne The vector OD takes DM as the upper limit of each component, adding one from the last digit
N=length(OD);
ND=OD;
if OD(N)<DM(N)
    ND(N)=ND(N)+1;
else
    if OD(N)>DM(N)
        error('Reache the upper limit. Can not add one')
    else
        if N==1
            error('Reache the upper limit. Can not add one')
        else
            ND(N)=1;
            ND(1:N-1)=AddOne(OD(1:N-1),DM(1:N-1));
        end
    end
end
end
function D=AddOnem2M(D,Dm,DM)
%AddOnem2M Vector OD takes Dm as the lower limit of each component and DM as the upper limit of each component, adding one from the last digit
if D(end)<DM(end)
    D(end)=D(end)+1;
else
    D(end)=Dm(end);
    D(1:end-1)=AddOnem2M(D(1:end-1),Dm(1:end-1),DM(1:end-1));
end
end