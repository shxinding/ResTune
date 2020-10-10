import numpy as np
import torch
import string
bandwidth=0.3
feature_tf_idf = {}
feature_tf_idf['sysbench'] =  [4.38454720e-01, 3.60586229e-04, 5.44811301e-04, 2.31115650e-01
, 1.15206641e-01, 7.47580877e-02, 5.62529920e-02, 3.83423935e-02
, 2.34946295e-02, 6.65937346e-03, 8.33289339e-03, 3.00440817e-03
, 1.50156113e-03, 8.62645482e-04, 1.10860740e-03, 0.00000000e+00]
feature_tf_idf['oltpbench'] = [6.85756209e-01, 3.63691059e-04, 5.49502401e-04, 1.30266870e-01
, 5.86378142e-02, 5.55244209e-02, 2.79038243e-02, 1.36983839e-02
, 6.58792922e-03, 6.36942438e-03, 8.10263436e-03, 2.85516974e-03
, 1.50709838e-03, 7.58875060e-04, 1.11815307e-03, 0.00000000e+00]

feature_tf_idf['twitter'] = feature_tf_idf['oltpbench']
feature_tf_idf['tpcc'] = [6.17135833e-01, 9.41270520e-05, 1.66702904e-04, 1.58347297e-01
, 8.11428425e-02, 3.99825736e-02, 3.82303090e-02, 1.96595737e-02
, 1.90909429e-02, 1.34756545e-02, 6.25688908e-03, 3.29102580e-03
, 1.53785964e-03, 9.62775892e-04, 4.18060239e-04, 2.07533232e-04]
feature_tf_idf['ic'] = [6.66611037e-01, 3.01715591e-07, 3.45622824e-06, 1.91454640e-01
, 6.09422607e-02, 2.42723787e-02, 2.77544012e-02, 6.85844170e-03
, 6.85362865e-03, 7.10183698e-03, 7.75103758e-03, 3.33058747e-04
, 5.32872420e-05, 4.76625278e-06, 1.97914273e-06, 3.48727391e-06]
feature_tf_idf['hotel'] = [6.18280355e-01, 2.45501282e-09, 1.38011733e-06, 2.05072338e-01
, 4.73947497e-02, 2.84933326e-02, 3.98513730e-02, 1.48392049e-02
, 1.99668987e-02, 1.34810420e-02, 1.02904131e-02, 1.84228855e-03
, 4.83739335e-04, 2.54933268e-07, 7.54783504e-09, 2.61965463e-06]
#Epanechnikov quadratic kernel

def compute_rank_weights_meta(base_models, workload_target, target_model):
    numDir = {}
    for task in list(base_models.keys()):
        if not task.split('_')[0] in numDir.keys():
            numDir[task.split('_')[0]] = 1
        else:
            numDir[task.split('_')[0]] = numDir[task.split('_')[0]] + 1

    weights = []
    for task in list(base_models.keys()):
        workload_source = task.split('_')[0].strip(string.digits)
        feature_source = np.array(feature_tf_idf[workload_source])
        feature_target = np.array(feature_tf_idf[workload_target])
        distance = np.linalg.norm(feature_source-feature_target)
        print (distance)
        t = distance / bandwidth
        if t < 1:
            gamma= 0.75 * (1-t*t)
        else:
            gamma = 0
        weights.append(gamma/numDir[task.split('_')[0]])
    if target_model:
        gamma_target = 0.75
        weights.append(gamma_target)
        weights = np.array(weights)
        weights = weights / weights.sum()
    weights = np.array(weights)
    rank_weights_sort = weights.tolist().copy()
    rank_weights_sort.sort(reverse=True)
    restrict_num = 9 if len(base_models) > 10 else -1
    weights[(weights < rank_weights_sort[restrict_num]).nonzero()] = 0
    sum_weight = float(weights.sum())
    weights = weights / sum_weight

    rank_weights_dir = {}
    count = 0
    for task in list(base_models.keys()):
        rank_weights_dir[task] = weights[count]
        count = count + 1
    if target_model:
        rank_weights_dir['target'] = weights[-1]
    weights = torch.tensor(weights)

    return rank_weights_dir, weights

if __name__ == "__main__":
    base_models = {}
    base_models['sysbench_1'] = 0
    base_models['tpcc_1'] = 0
    base_models['twitter'] = 0
    for workload_target in ['sysbench', 'tpcc', 'oltpbench']:
        rank_weight_dir, rank_weights = compute_rank_weights_meta(base_models, workload_target, 1)
        d_order = sorted(rank_weight_dir.items(), key=lambda x: x[1], reverse=True)
        map_num = sum([w[1] != 0 for w in d_order])
        print ("[Wokrload Mapping]: map {} workloads, top weights: {}".format(map_num, d_order[:map_num]))

