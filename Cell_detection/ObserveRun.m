close all;
load('ExpDir\inter_valid.mat')
for i = 1:1000
L = label(i,:,:);
if ~any(L(:))
break;
end
end
figure, imagesc(permute(input(i,:,:,:), [2,3,4,1]));
figure, imagesc(permute(label(i,:,:), [2,3,1]));
figure, imagesc(permute(logits(i,:,:).*(logits(i,:,:)>0.05), [2,3,1]));
impixelinfo
%%
% close all;
load('ExpDir\inter_valid.mat')
for i = 1:1000
L = label(i,:,:);
if any(L(:))
break;
end
end
figure, imagesc(permute(input(i,:,:,:), [2,3,4,1]));
figure, imagesc(permute(label(i,:,:), [2,3,1]));
figure, imagesc(permute(logits(i,:,:).*(logits(i,:,:)>0.05), [2,3,1]));
impixelinfo
%%
load('ExpDir\inter_valid.mat')
sum = 0;
for i = 1:1000
    L = label(i,:,:);
    Lo = logits(i,:,:,:)>0.01;
    sum = sum + double(any(L(:))==any(Lo(:)));
end
display(sum/1000);
%%
load('ExpDir\avg_training_loss.mat')
figure, plot(avg_training_loss, 1:length(avg_training_loss), avg_validation_loss, 1:length(avg_validation_loss))
