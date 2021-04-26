import os
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from copy import deepcopy
from sklearn.metrics import auc
from skimage import morphology, graph
import tensorflow as tf
from random import randint

def get_my_metrics(preds,masks,fovs):

    auc_list = []
    acc_list = []
    sen_list = []
    spe_list = []
    accs_list = [0]*11
    sens_list = [0]*11
    spes_list = [0]*11

    for i in range(preds.shape[0]):

        auc, acc, sen, spe, accs, sens, spes = my_metrics(masks[i,:,:], preds[i,:,:], fovs[i,:,:])

        for j, elem in enumerate(accs):
            accs_list[j] += elem

        for j, elem in enumerate(sens):
            sens_list[j] += elem

        for j, elem in enumerate(spes):
            spes_list[j] += elem

        auc_list.append(auc)
        acc_list.append(acc)
        sen_list.append(sen)
        spe_list.append(spe)

    return np.mean(auc_list), np.mean(acc_list), np.mean(sen_list), np.mean(spe_list)

def my_metrics(y_true, y_pred, fov):

    fpr = []
    tpr = []
    accs = []
    sens = []
    spes = []

    for i in range(100,-1,-1):

        thresh = 0.01*i
        this_pred = np.zeros(y_pred.shape)
        this_pred[np.where(y_pred>=thresh)] = 1

        TP = np.sum(np.multiply(np.multiply(y_true, this_pred), fov))
        TN = np.sum(np.multiply(np.multiply(1-y_true, 1-this_pred),fov))
        FP = np.sum(np.multiply(np.multiply(1-y_true, this_pred), fov))
        FN = np.sum(np.multiply(np.multiply(y_true, 1-this_pred), fov))
        
        fpr.append(FP/(TN+FP))
        tpr.append(TP/(TP+FN))

        if i==50:
            acc = (TP+TN) / (TP+TN+FP+FN)
            sen = TP / (TP+FN)
            spe = TN / (TN+FP)

        if i%10==0:
            accs.append((TP+TN) / (TP+TN+FP+FN))
            sens.append(TP / (TP+FN))
            spes.append(TN / (TN+FP))

    fpr.append(1)
    tpr.append(1)

    return auc(fpr, tpr), acc, sen, spe, accs, sens, spes

def get_mix_coef(init_alpha=0.99, decay=0.005, start_epoch = 0, min_alpha = 0.3, epoch=None):

    if epoch <= start_epoch:
        return init_alpha

    else:
        alpha = init_alpha - (epoch-start_epoch)*decay

        if alpha < min_alpha:
            return min_alpha
        else:
            return alpha


def log_gaussian(x, mu, logvar):

    PI = tf.constant([np.pi])
    
    x = tf.reshape(x,shape=[x.shape[0],-1])
    mu = tf.reshape(mu,shape = [x.shapep[0],-1])
    logvar = tf.reshape(logvar,shape = [x.shape[0],-1])
    
    N, D = x.shape

    log_norm = (-1/2) * (D * tf.math.log(2*PI) + 
                         tf.math.reduce_sum(logvar,axis = 1) +
                         tf.math.reduce_sum((((x-mu)**2)/(tf.keras.activations.exponential(logvar))),axis = 1))

    return log_norm



def topo_metric(gt, pred, thresh=0.5, n_paths=1000):

    # 0, 1 and 2 mean, respectively, that path is infeasible, shorter/larger and correct
    result = []

    # binarize pred according to thresh
    pred_bw = (pred>thresh).astype(int)
    pred_cc = morphology.label(pred_bw)

    # get centerlines of gt and pred
    gt_cent = morphology.skeletonize(gt>0.5)
    gt_cent_cc = morphology.label(gt_cent)
    pred_cent = morphology.skeletonize(pred_bw)
    pred_cent_cc = morphology.label(pred_cent)

    # costs matrices
    gt_cost = np.ones(gt_cent.shape)
    gt_cost[gt_cent==0] = 10000
    pred_cost = np.ones(pred_cent.shape)
    pred_cost[pred_cent==0] = 10000
    
    # build graph and find shortest paths
    for i in range(n_paths):

        # pick randomly a first point in the centerline
        R_gt_cent, C_gt_cent = np.where(gt_cent==1)
        idx1 = randint(0, len(R_gt_cent)-1)
        label = gt_cent_cc[R_gt_cent[idx1], C_gt_cent[idx1]]
        ptx1 = (R_gt_cent[idx1], C_gt_cent[idx1])

        # pick a second point that is connected to the first one
        R_gt_cent_label, C_gt_cent_label = np.where(gt_cent_cc==label)
        idx2 = randint(0, len(R_gt_cent_label)-1)
        ptx2 = (R_gt_cent_label[idx2], C_gt_cent_label[idx2])

        # if points have different labels in pred image, no path is feasible
        if (pred_cc[ptx1] != pred_cc[ptx2]) or pred_cc[ptx1]==0:
            result.append(0)

        else:
            # find corresponding centerline points in pred centerlines
            R_pred_cent, C_pred_cent = np.where(pred_cent==1)
            poss_corr = np.zeros((len(R_pred_cent),2))
            poss_corr[:,0] = R_pred_cent
            poss_corr[:,1] = C_pred_cent
            poss_corr = np.transpose(np.asarray([R_pred_cent, C_pred_cent]))
            dist2_ptx1 = np.sum((poss_corr-np.asarray(ptx1))**2, axis=1)
            dist2_ptx2 = np.sum((poss_corr-np.asarray(ptx2))**2, axis=1)
            corr1 = poss_corr[np.argmin(dist2_ptx1)]
            corr2 = poss_corr[np.argmin(dist2_ptx2)]
            
            # find shortest path in gt and pred
            
            gt_path, cost1 = graph.route_through_array(gt_cost, ptx1, ptx2)
            gt_path = np.asarray(gt_path)

            pred_path, cost2 = graph.route_through_array(pred_cost, corr1, corr2)
            pred_path = np.asarray(pred_path)


            # compare paths length
            path_gt_length = np.sum(np.sqrt(np.sum(np.diff(gt_path, axis=0)**2, axis=1)))
            path_pred_length = np.sum(np.sqrt(np.sum(np.diff(pred_path, axis=0)**2, axis=1)))
            if pred_path.shape[0]<2:
                result.append(2)
            else:
                if ((path_gt_length / path_pred_length) < 0.9) or ((path_gt_length / path_pred_length) > 1.1):
                    result.append(1)
                else:
                    result.append(2)

    return result.count(0), result.count(1), result.count(2)
