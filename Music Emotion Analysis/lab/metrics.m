function [ acc, prec, rec, f1 ] = metrics( real_val, classif_out)

true_pos = sum((classif_out == 1) & (real_val == 1));
false_pos = sum((classif_out == 1) & (real_val == -1));
false_neg = sum((classif_out == -1) & (real_val == 1));
true_neg = sum((classif_out == -1) & (real_val == -1));

acc = (true_pos+true_neg)/(true_pos+true_neg+false_pos+false_neg);
prec = true_pos / (true_pos+false_pos);
rec = true_pos / (true_pos + false_neg);
f1 = 2*(prec*rec)/(prec+rec);

end

