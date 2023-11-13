import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score,accuracy_score
from scipy.special import softmax

#Not Used 
def plot_grad_flow(logger, named_parameters, epoch):
    """Plots the gradients flowing through different layers in the net during training.
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    grad_dict = {}
    max_grad_dict = {}
    grad_dict_raw = {}
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n) and ("classifier" in n):
            if not p.grad is None:
                logger.add_histogram(n, p.grad, epoch)

#Not Used 
def make_onehot(vec, num_classes):
    if num_classes ==1: 
        return vec.reshape((-1,1))
    vec = np.array(vec)
    vec= vec.astype('int')
    one_hot = np.zeros((vec.shape[0], num_classes))
    # here i just loop through the array.  since each label is  a zero, 1 or two.  i just index the one hot vector array
    
    for i, e in enumerate(vec):
        one_hot[i, e] = 1
    return one_hot

#Not Used 
def get_best_t(y_true,scores):
    fpr, tpr, thresholds = metrics.roc_curve(y_true,scores)
    arr  = np.array((fpr,tpr)).T
    ideal_idx = np. sum( (arr - np.array((0,1)) )**2 ,axis=1).argmin()
    t = thresholds[ideal_idx]
    ideal_point = (fpr[ideal_idx],tpr[ideal_idx])
    return t ,ideal_point


def calculate_metrics(tru_labels: np.array, out_probs: np.array, num_classes,prefix="",class_names=None):
    # note: true labels is of shape [nx1] and each finding can range from  0 to C-1. C being num classes
    tru_labels = np.array(tru_labels)
    #out_probs = softmax(np.array(out_probs), axis=1)
    out_labels = out_probs.argmax(axis=1)
    interest = range(0, num_classes) 
    metrics = {"precision": precision_score, "recall": recall_score,"acc":accuracy_score}
    val_dict = dict()
    for cl in interest:
        for k in metrics:
            func = metrics[k]
            key = f"{prefix}_{class_names[cl]}_{k}" if prefix else  f"{cl}_{k}"
            val_dict[key] = func(tru_labels==cl, out_labels==cl )
    for e in range(num_classes):
        try:
            auc_name = f"{prefix}_{1}_auc" if prefix else f"{e}_auc" 
            if  prefix: 
                val_dict[auc_name] = roc_auc_score(
                    (tru_labels == 1).astype(int), out_probs
                )
            else: 
                val_dict[auc_name] = roc_auc_score(
                (tru_labels == e).astype(int), out_probs[:, e]
                )
        except ValueError:
            val_dict[auc_name] = 0.5
    return val_dict
