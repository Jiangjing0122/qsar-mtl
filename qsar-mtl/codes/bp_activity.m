function f1_mean = bp_activity(data, indices, fold, g)

for f=1:fold
    test_data=(indices==f);
    train_data=~test_data;
    
    input_train=data(train_data,1:end-1)';
    trainY=data(train_data,end);
    
    output_train=zeros(size(trainY,1),2);
    for i=1:size(trainY,1)
        switch trainY(i)
            case 1
                output_train(i,:)=[1 0];
            case -1
                output_train(i,:)=[0 1];
        end
    end
    output_train=output_train';
    
    input_test=data(test_data,1:end-1)';
    
    output_test=data(test_data,end)';
    
    % the net architecture initialization
    innum=size(input_train,1);
    midnum=25;
    outnum=2;
    
    % the weights initialization
    w1=rands(midnum,innum);
    b1=rands(midnum,1);
    w2=rands(midnum,outnum);
    b2=rands(outnum,1);
    
    w2_1=w2;w2_2=w2_1;
    w1_1=w1;w1_2=w1_1;
    b1_1=b1;b1_2=b1_1;
    b2_1=b2;b2_2=b2_1;
    
    %the learning rate, which needs be optimized by different data
    xite=0.05;
    alfa=0.01;
    loopNumber=500;
    
    I=zeros(1,midnum);
    Iout=zeros(1,midnum);
    FI=zeros(1,midnum);
    dw1=zeros(innum,midnum);
    db1=zeros(1,midnum);
    
    %% traing
    E=zeros(1,loopNumber);
    for ii=1:loopNumber
        E(ii)=0;
        for i=1:1:size(input_train,2)
            x=input_train(:,i);
            for j=1:1:midnum
                I(j)=input_train(:,i)'*w1(j,:)'+b1(j);
                Iout(j)=1/(1+exp(-I(j)));
            end
            yn=w2'*Iout'+b2;
            
            %error calculation
            e=output_train(:,i)-yn;
            E(ii)=E(ii)+sum(abs(e));
            
            dw2=e*Iout;
            db2=e';
            
            for j=1:1:midnum
                S=1/(1+exp(-I(j)));
                FI(j)=S*(1-S);
            end
            for k=1:1:innum
                for j=1:1:midnum
                    dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));
                    db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));
                end
            end
            
            w1=w1_1+xite*dw1';
            b1=b1_1+xite*db1';
            w2=w2_1+xite*dw2';
            b2=b2_1+xite*db2';
            
            w1_2=w1_1;w1_1=w1;
            w2_2=w2_1;w2_1=w2;
            b1_2=b1_1;b1_1=b1;
            b2_2=b2_1;b2_1=b2;
        end
    end
    
    
    %% prediction
    testNum=size(output_test,2);
    fore=zeros(2,testNum);
    
    for i=1:testNum
        for j=1:1:midnum
            I(j)=input_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        fore(:,i)=w2'*Iout'+b2;
    end
    
    % ouput claculation
    output_fore=zeros(1,testNum);
    for i=1:testNum
        if fore(1,i)>=fore(2,i)
            output_fore(i)=1;
        else
            output_fore(i)=-1;
        end
    end
    
    %calculate f1 values
    f1_value(f)=eval_f1(output_test,output_fore);
end
f1_value=rmmissing(f1_value);
f1_mean=mean(f1_value);
end