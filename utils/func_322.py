import os
import copy
import numpy as np
from torch import nn
from sklearn import metrics
import matplotlib.pyplot as plt

# Plot function to plot the losses and save fig
def plot_figure(loss_list, label_name, model_id, output_dir_loss):
    plt.plot(loss_list, label=label_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_id)
    plt.legend()
    plt.savefig(os.path.join(output_dir_loss, label_name + '.png'))
    plt.show()

def save_figures(train_losses, valid_losses, output_dir_loss, total_epochs):
    x_axis = np.arange(1, total_epochs+1)
    # Plot train loss
    plt.figure(figsize=(20, 10))
    plt.plot(x_axis, train_losses, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    train_loss_path = os.path.join(output_dir_loss, 'train_loss.png')
    plt.savefig(train_loss_path)
    plt.clf()  # Clear the figure for the next plot

    # Plot validation loss
    plt.figure(figsize=(20, 10))
    plt.plot(x_axis, valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()
    valid_loss_path = os.path.join(output_dir_loss, 'valid_loss.png')
    plt.savefig(valid_loss_path)
    plt.clf()  # Clear the figure for the next plot

    # Plot combined losses
    plt.figure(figsize=(20, 10))
    plt.plot(x_axis,train_losses, label='Train Loss')
    plt.plot(x_axis, valid_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Combined Losses')
    plt.legend()
    combined_loss_path = os.path.join(output_dir_loss, 'combined_loss.png')
    plt.savefig(combined_loss_path)
    plt.clf()  # Clear the figure for the next plot

# Check if the recieving loss is the smallest
def is_smaller(number, number_list):
    smallest = number_list[0]
    for num in number_list:
        if num < smallest:
            smallest = num
    return number == smallest

def is_larger(number, number_list):
    largest = number_list[0]
    for num in number_list:
        if num > largest:
            largest = num
    return number == largest

def list_accuracy(list1, list2):
    # Ensure that both lists have the same length
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length")

    # Initialize a counter for correct elements
    correct_count = 0

    # Iterate through the elements of both lists and compare them
    for elem1, elem2 in zip(list1, list2):
        if elem1 == elem2:
            correct_count += 1

    # Calculate accuracy as the percentage of correct elements
    accuracy = (correct_count / len(list1))

    return accuracy

def print_sub_accuracy(c_cls, r_cls):
    am_accuracy = list_accuracy(c_cls[0:8], r_cls[0:8]) 
    dc_accuracy = list_accuracy(c_cls[8:13], r_cls[8:13])
    okc_accuracy = list_accuracy(c_cls[13:23], r_cls[13:23])
    rc_accuracy = list_accuracy(c_cls[23:], r_cls[23:])

    print(f"AM: {am_accuracy:.4f}, DC: {dc_accuracy:.4f}, OKC: {okc_accuracy:.4f}, RC: {rc_accuracy:.4f}")
    
    # print("Ground Truth:", end="  ")
    # print(c_cls)
    # print("Predict Label:")
    # print("AM:", end="  ")
    # print(r_cls[0:8])
    # print("DC:", end="  ")
    # print(r_cls[8:13])
    # print("OKC:", end=" ")
    # print(r_cls[13:23])
    # print("RC:", end="  ")
    # print(r_cls[23:])

    return

def list_from_label(tensor_a, tensor_b):
    a1 = tensor_a[:, :2]
    a2 = tensor_a[:, 2:4]
    a3 = tensor_a[:, 4:]
    a_list = [a1, a2, a3]

    b1 = tensor_b[:, :2]
    b2 = tensor_b[:, 2:4]
    b3 = tensor_b[:, 4:]
    b_list = [b1, b2, b3]
    
    return a_list, b_list

def cal_atr_list(o_list, c_list, tensor_length):
    # print(o_list)
    # print(c_list)
    out_list = [0]*tensor_length
    divided_ol = [[]]*tensor_length
    divided_cl = [[]]*tensor_length

    for i in range(tensor_length):
        for o, c in zip(o_list, c_list):
            divided_ol[i].append(o[0][i].cpu())
            divided_cl[i].append(c[0][i].cpu())
    
    precision = 0
    f1 = 0
    for i in range(tensor_length):
        p, r, f = calculate_metrics(divided_cl[i], divided_ol[i])
        precision += p
        f1 += f
    
    count = 0
    for o, c in zip(o_list, c_list):
        count += 1
        for i in range(tensor_length):
            if o[0,i] == c[0,i]:
                out_list[i] += 1
    
    return [round(value / count, 4) for value in out_list], precision/tensor_length, f1/tensor_length

def cal_atr_list_2(o_list, c_list, tensor_length):
    # print(o_list)
    # print(c_list)
    out_list = [0]*tensor_length
    divided_ol = [[]]*tensor_length
    divided_cl = [[]]*tensor_length

    for i in range(tensor_length):
        for o, c in zip(o_list, c_list):
            divided_ol[i].append(o[i])
            divided_cl[i].append(c[i])
    
    precision = 0
    f1 = 0
    for i in range(tensor_length):
        p, r, f = calculate_metrics(divided_cl[i], divided_ol[i])
        precision += p
        f1 += f
    
    count = 0
    for o, c in zip(o_list, c_list):
        count += 1
        for i in range(tensor_length):
            if o[i] == c[i]:
                out_list[i] += 1
    
    return [round(value / count, 4) for value in out_list], precision/tensor_length, f1/tensor_length

def calculate_metrics(c_list, p_list):
    precision = metrics.precision_score(c_list, p_list, average='weighted')
    recall = metrics.recall_score(c_list, p_list, average='weighted')
    f1 = metrics.f1_score(c_list, p_list, average='weighted')
    # auc = roc_auc_score(c_list, p_list, average='macro')

    return precision, recall, f1

class Dice_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Dice_Loss, self).__init__()

    def forward(self, inputs, targets, smooth=0):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        loss = 1-dice
        
        return loss

def metric_test(gt,pred):
    preds = pred.detach().numpy()
    gts = gt.detach().numpy()
    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    fp_array = copy.deepcopy(pred)  # keep pred unchanged
    fn_array = copy.deepcopy(gdth)
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    union = gdth | pred
    intersection_sum = np.count_nonzero(intersection)
    union_sum = np.count_nonzero(union)

    tp_array = intersection

    tmp = pred - gdth
    fp_array[tmp < 1] = 0

    tmp2 = gdth - pred
    fn_array[tmp2 < 1] = 0

    tn_array = np.ones(gdth.shape) - union

    tp, fp, fn, tn = np.sum(tp_array), np.sum(fp_array), np.sum(fn_array), np.sum(tn_array)

    smooth = 0.001
    precision = tp / (pred_sum + smooth)
    recall = tp / (gdth_sum + smooth)

    false_positive_rate = fp / (fp + tn + smooth)
    true_positive_rate = tp/ (tp + fn + smooth)
    false_negtive_rate = fn / (fn + tp + smooth)

    size = 1
    for i in gdth.shape:
        size = size*i

    jaccard = (intersection_sum + smooth) / (union_sum + smooth)
    dice = (2 * intersection_sum + smooth) / (gdth_sum + pred_sum + smooth)


    if gdth_sum == pred_sum == 0:
        subtract = 1
        lesion_jaccard = 0
        lesion_dice = 0
    else:
        subtract = 0
        lesion_jaccard = (intersection_sum) / (union_sum)
        lesion_dice = (2 * intersection_sum) / (gdth_sum + pred_sum)

    fpr, tpr, thresholds = metrics.roc_curve(gdth.reshape(size),
                                             pred.reshape(size),
                                             pos_label=1)
    auc = metrics.auc(fpr, tpr)


    # return precision,false_positive_rate,false_negtive_rate,dice,jaccard,lesion_dice,lesion_jaccard,subtract,auc,intersection_sum,gdth_sum,pred_sum,union_sum
    return precision,dice,jaccard,auc

def metric_valid(gt,pred):
    preds = pred.detach().numpy()
    gts = gt.detach().numpy()
    pred = preds.astype(int)  # float data does not support bit_and and bit_or
    gdth = gts.astype(int)  # float data does not support bit_and and bit_or
    gdth_sum = np.sum(gdth)
    pred_sum = np.sum(pred)
    intersection = gdth & pred
    intersection_sum = np.count_nonzero(intersection)

    smooth = 0.001
    dice = (2 * intersection_sum + smooth) / (gdth_sum + pred_sum + smooth)

    return dice