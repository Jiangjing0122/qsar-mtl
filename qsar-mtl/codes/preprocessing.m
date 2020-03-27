clear
clc

fileID = fopen('data\random_group.txt');
dataCell = textscan(fileID,'','Delimiter','	','TreatAsEmpty',{'na'});
data=cell2mat(dataCell);
% dataFilled= fillmissing(data,'previous');
len=size(data,1);
dataFilled= fillmissing(data,'movmean',len);

% dataFilled=dataFilled(:,1:100);


% group = [157,50,55,56]; %group1
% group = [250,61,66,64]; %group2
% group = [59,111,112]; %group3
% group = [489,488,492]; %group4
% group = [93,75,22]; %group5
% group = [1212,789,51]; %group6
group = [50,250,59,489,75,51]; %random group

labels=[];
for i=1:length(group)
    a=ones(1,group(i));
    b=zeros(1,group(i));
    c=[a,b];
    labels=[labels,c];
end

data=[dataFilled,labels'];

save 'data\random_group' data