classdef MyOctree
    %MYOCTREE  A class to accelerate the neighborhood search in MLS
    properties
        points  % Store all points as column vectors
        Mbox  % The upper and lower limits of each component are stored in the form of D*2
        dB       % A d-dimensional edge vector 
        dBm    % The d-dimensional edge vector of the hyper-cuboid in deepest level
        MD      % Deepest level
        D         % Dimension
        Maps   % A tensor which stores the index of the index vector set
        idxs      % The index vector set
        R         %The minimum diameter of a leaf node
        Nnum  %Number of neighbors for a inside node
        dN       %The degree of offset of the neighbor for the inside node is a matrix of D* (3^D)
    end
    methods
        function obj = MyOctree(X,box,R)
            %Constructor build the G-Octree in paper
            %% Initialization
            obj.points=X;
            obj.Mbox=box;
            obj.dBm=(box(:,2)-box(:,1));
            obj.dB=obj.dBm;
            dm=min(obj.dB);
            MD=1;
            while 1
                dm=dm/2;
                if dm>R
                    MD=MD+1;
                else
                    break;
                end
            end
            obj.MD=MD;
            obj.R=dm*2;
            [obj.D,N]=size(X);
            Cn=2^obj.D;                                           %The number of children
            MN=Cn^(MD-1);                                    %Number of leaf nodes
            Dnmax=2^(MD-1)*ones(obj.D,1);
            obj.Maps=Dtensor(Dnmax);                    %The pre-stored number 
            obj.idxs=cell(MN,1);
            Di=ones(obj.D,1);                                    %Level 1
            obj.idxs{obj.Maps.GetData(Di)}=1:1:N;     %The indexs stored in root
            Dnold=ones(obj.D,1);
            Ddold=Dnmax(1);
            for i=2:obj.MD                                        %Level by Level, store the index of indexs into Maps
                Dnnew=Dnold*2;
                Ddnew=Ddold/2;                                %The edge vector in next level 
                Nold=Cn^(i-2);
                obj.dBm=obj.dBm./2;
                Dj=ones(obj.D,1);
                Dj(end)=0;
                for j=1:Nold                                        
                Dj=AddOne(Dj,Dnold);  
                Mb=box(:,1)+(2*Dj-1).*obj.dBm; 
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
        function Dk= SearchBox(obj,point)
            %SearchBox Search for the leaf node that a point belongs to in the tree
            %   point is the obj.D-dimensional column vector
            %   Dk represents the coordinates of the leaf node in the obj.D-dimensional tensor and is the obj.D-dimensional column vector
            %   Db is the edge vector of a leaf node 
            Dk=ceil((point-obj.Mbox(:,1))./obj.dBm);
            Dk(Dk<=0)=1;
        end
        function Idx= RangeSearch(obj,point,r)
            %RangeSearch Search for all idxs of the data point in B(obj,point ,r)
            %   point is the obj.D-dimensional column vector
            %   r must be less than the diameter of the leaf node obj.R
            %   Idx is a row vector
            if r>obj.R
                warning('It's out of range. Results may be inaccurate')
            end 
            [~,N]=size(obj.points);
            IB=false(1,N);
            Dk= SearchBox(obj,point);
            Bl=obj.Mbox(:,1)+(Dk-1).*obj.dBm;
            Dl=point-Bl;
            Dr=obj.dBm-Dl;
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
        error('Reache the upper limit. Can't add one')
    else
        if N==1
            error('Reache the upper limit. Can't add one')
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