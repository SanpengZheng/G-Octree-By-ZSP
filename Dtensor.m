classdef Dtensor
    % Dtensor A self-defining D-dimensional tensor in the form of a class
    %   The D-dimensional tensor is produced from the input D-dimensional vector, initially all zeros, written and read using the write and read functions

    properties(SetAccess=private)
        T
        D
        L    %Length
        SN %Each position represents the number, in decimal
    end

    methods
        function obj = Dtensor(D)
            obj.L=length(D);
            obj.SN=zeros(obj.L,1);
            obj.SN(1)=1;
            for i=2:obj.L
                obj.SN(i)=D(i)*obj.SN(i-1);
            end
            N=obj.SN(obj.L)*D(obj.L);
            %obj.T = reshape(zeros(N,1),D');
            obj.T = reshape(1:N,D');
            obj.D=D;
        end
        function obj = SetData(obj,Dk,inputArg)
            %write 
            %   Write the input to the corresponding location of Dk
            for i=1:obj.L
                if Dk(i)>obj.D(i)
                    error('The input vector exceeds the upper bound')
                end
            end
            obj.T(1+(Dk-1)'*obj.SN) = inputArg;
        end
        function data = GetData(obj,Dk)
            %read 
            %   Read the data corresponding to the position of obj and Dk
            for i=1:obj.L
                if Dk(i)>obj.D(i)
                    error('The input vector exceeds the upper bound')
                end
            end
            data = obj.T(1+(Dk-1)'*obj.SN);
        end
    end
end