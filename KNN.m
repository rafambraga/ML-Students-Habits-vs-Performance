%% Data Mining
% Rafael Marques Braga

clear, clc

% Reading file
filename = 'knn_bdf_train.csv';
data = table2array(readtable(filename));

% Normalizing data

A = normalize(data, 'range'); 

% Reading file
filename = 'knn_bdf_test.csv';
data2 = table2array(readtable(filename));

% Normalizing data

test_data = normalize(data2, 'range');

%% Code
% Important Values:  
n_att = 27 
n_ex = 519;
class = n_att + 1;
pos_val = 1;                            % Cannot be 0!!!!!
att1 = 1;
att2 = 2;
att3 = 3;
major_class = 0;

n_ex_test = 130;
n_att_test = n_att;

Ac = A;

%% Finding Distance and majority class undersampling
counter = 1;
t_removed = 0;
while (counter ~=0)
    counter = 0;
    p = 1:n_att;
    for y = 1:n_ex
        for x = 1:n_ex
            dis2(y,x) = sqrt(sum((A(y,p)-A(x,p)).^2));
        end
    end
    

    dis2(dis2 == 0) = inf;
    [b,i] = min(dis2);

    % Tomek links majority class undersampling

    for ref = 1:n_ex
        nn = i(ref);
        if (i(nn) == ref)
            tm(ref) = 1;
            if(A(ref, n_att+1)~=A(nn, n_att+1))
                if(A(nn,n_att+1)==major_class)                        % Negative class
                    A(nn,:) = inf;
                    t_removed = t_removed + 1;
                    counter = counter + 1; 
                end
            end
        else 
            tm(ref) = 0;
        end
    end


end
X = [num2str(t_removed),' examples were removed from the training data (Tomek Links) '];
disp(X)        
%% Near K
%Finds distance matrix    
v = 1:n_att_test;
for j = 1:n_ex_test
    for k = 1:n_ex
        dist(k,j) = sqrt(sum((test_data(j,v)-A(k,v)).^2));          % Change here
    end
end

[l1,l2] = min(dist);    

% ------------
sort_dist = sort(dist,'ascend');

count = zeros(1, n_ex_test);
h_class = zeros(1, n_ex_test);
tp = 0;
fp = 0;
tn = 0;
fn = 0;
for p1 = 1:n_ex_test
    for count1 = 1:3
        min_v = sort_dist(count1,p1);
        extracted = dist(:,p1);
        minl = find(extracted==min_v);
        if(A(minl,class)==pos_val)                                 %%change here
            count(p1) = count(p1) +1;
        end
    end
    if(count(p1)>1)
        h_class(p1) = pos_val;
    end
    if(h_class(p1)==pos_val) && (test_data(p1,class)==pos_val)
        tp = tp +1;
    elseif(h_class(p1)==pos_val) && (test_data(p1,class)~=pos_val)
        fp = fp + 1;
    elseif(h_class(p1)~=pos_val) && (test_data(p1,class)~=pos_val)
        tn = tn + 1;
    else
        fn = fn + 1;
    end
end
S0 = ['# of True Positives: ', num2str(tp)];
S1 = ['# of False Positives: ', num2str(fp)];
S2 = ['# of True Negatives: ', num2str(tn)];
S3 = ['# of False Negatives: ', num2str(fn)];
disp(S0)
disp(S1)
disp(S2)
disp(S3)
precision = tp / (tp +fp);
recall = tp / (tp + fn);
Se = tp/(tp+fn);
Sp = tn/(tn+fp);
EE = (fp + fn)/(fp + fn + tp + tn);
Acc = 1 - EE;

Fi0 = ['Precision = ', num2str(precision)];
Fi1 = ['Recall = ', num2str(recall)];
Fi2 = ['Sensivity = ', num2str(Se)];
Fi3 = ['Specificity = ', num2str(Sp)];
Fi4 = ['Error rate = ', num2str(EE)];
Fi5 = ['Accuracy = ', num2str(Acc)];
disp(Fi0)
disp(Fi1)
disp(Fi2)
disp(Fi3)
disp(Fi4)
disp(Fi5)