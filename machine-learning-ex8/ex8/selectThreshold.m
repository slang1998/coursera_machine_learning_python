function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% 가장 낮은 확률에서 높은 확률값까지 1000 단계로 구분한다.
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

	% 엡실론보다 작은 것은 anomaly로 구분해서 벡터로 저장한다.
	predictions = (pval < epsilon);

	% true positives 계산: 실제 anomaly이고 anomaly로 예측함
	tp = sum((predictions == 1) & (yval == 1));

	% false positives 계산: 실제 anomaly가 아닌데 anomaly로 예측함
	fp = sum((predictions == 1) & (yval == 0));

	% false negatives 계산: 실제 anomaly인데 anomaly가 아님으로 예측함
	fn = sum((predictions == 0) & (yval == 1));

	% 정확도 계산: 
	prec = tp / (tp + fp);

	% 재현율 계산: 
	rec = tp / (tp + fn);

	% F1 score 계산
	F1 = 2 * prec * rec / (prec + rec);


    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
