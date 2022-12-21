import os
import itertools
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, precision_score, f1_score, recall_score, confusion_matrix, \
    balanced_accuracy_score, accuracy_score

from utils.constants import class_names


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def AverageList(lst):
    """ Compute average of a list"""
    return sum(lst) / len(lst)


def save_clf_metrics(targets, preds, path, dataset):
    """
    saves the classification metrics on all the test cities
    Args:
        targets: ground truth labels
        preds: predictions by the model
        path: path to the log folders
        dataset: name of the data set partition with which log file name would be saved, usually set as 'test'
    """

    log_file = os.path.join(path, path.split(os.sep)[-1]+'_' + dataset + '.txt')
    y_targ = targets.detach().numpy()
    y_pred = preds.detach().numpy()
    # calc own metrics
    simple_distance = calc_distance(y_targ, y_pred)
    squared_distance = calc_distance(y_targ, y_pred, square = True)
    # calc std metrics
    val_f1_mi = round(f1_score(y_targ, y_pred, average='micro'), 4)
    val_recall_mi = round(recall_score(y_targ, y_pred, average='micro'), 4)
    val_precis_mi = round(precision_score(y_targ, y_pred, average='micro'), 4)
    val_f1 = round(f1_score(y_targ, y_pred, average='macro'), 4)
    val_recall = round(recall_score(y_targ, y_pred, average='macro'), 4)
    val_precis = round(precision_score(y_targ, y_pred, average='macro'), 4)

    val_macd = np.mean(np.abs(y_targ - y_pred))
    val_cm = confusion_matrix(y_targ, y_pred)
    accuracy = accuracy_score(y_targ, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_targ, y_pred)
    classwise_accuracy = confusion_matrix(y_targ, y_pred, normalize="true").diagonal()
    print(classification_report(y_targ, y_pred))
    print("\n — test_f1_macro: {} \n — test_f1_micro: {} \n — test_precis_macro: {} \n — test_precis_micro: {} \n "
          "— test_recall_macro: {} \n — test_recall_micro: {} \n — test_macd: {} \n — test_accuracy: {} "
          "\n — test_balanced_accuracy: {} \n — class wise accuracy: {}".format(val_f1, val_f1_mi, val_precis,
                                                                                      val_precis_mi,
                                                                                      val_recall,
                                                                                      val_recall_mi,
                                                                                      val_macd, accuracy,
                                                                                      balanced_accuracy,
                                                                                      classwise_accuracy))
    plot_confusion_matrix(val_cm, class_names, log_file.replace('.txt', '_cm.png'))
    plot_normalized_confusion_matrix(val_cm, class_names, log_file.replace('.txt', '_cmnorm.png'))

    with open(log_file, 'a') as f:
        f.writelines(
            "\n — test_f1_macro: {} \n — test_f1_micro: {} \n — test_precis_macro: {} \n — test_precis_micro: {} \n "
            "— test_recall_macro: {} \n — test_recall_micro: {} \n — test_macd: {} \n — test_accuracy: {}"
            "\n — test_balanced_accuracy: {} \n — class wise accuracy: {}\n — normal class distance: {}\n "
            "— squared class distance: {}".format(val_f1, val_f1_mi, val_precis, val_precis_mi, val_recall,
                                                  val_recall_mi, val_macd, accuracy, balanced_accuracy,
                                                  classwise_accuracy, simple_distance, squared_distance))


def save_clf_metrics_citywise(targets, preds, path, ID_list, dataset_name):
    """
    saves the classification metrics on indvidual test cities
    Args:
        targets: ground truth labels
        preds: predictions by the model
        path: path to the log folders
        dataset_name: name of the data set partition with which log file name would be saved, usually set as 'test'
    Returns a csv file for a city with grid id, ground truth label and predicted label
    """

    log_file = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '.txt')
    y_targ = targets.detach().numpy()
    y_pred = preds.detach().numpy()

    # calc own metrics
    simple_distance = calc_distance(y_targ, y_pred)
    squared_distance = calc_distance(y_targ, y_pred, square=True)

    # calc std metrics
    val_f1_mi = round(f1_score(y_targ, y_pred, average='micro'), 4)
    val_recall_mi = round(recall_score(y_targ, y_pred, average='micro'), 4)
    val_precis_mi = round(precision_score(y_targ, y_pred, average='micro'), 4)
    val_f1 = round(f1_score(y_targ, y_pred, average='macro'), 4)
    val_recall = round(recall_score(y_targ, y_pred, average='macro'), 4)
    val_precis = round(precision_score(y_targ, y_pred, average='macro'), 4)

    val_macd = np.mean(np.abs(y_targ - y_pred))
    val_cm = confusion_matrix(y_targ, y_pred)
    accuracy = accuracy_score(y_targ, y_pred)

    for count_index in range(0, val_cm.shape[0]):
        if val_cm.sum(axis=1)[count_index] != 0:
            norm_val_cm_row = np.around(val_cm.astype('float')[count_index] / val_cm.sum(axis=1)[count_index],
                                        decimals=2)
        else:
            norm_val_cm_row = [0.0 for col in range(val_cm.shape[1])]
        if count_index == 0:
            norm_val_cm = norm_val_cm_row
        else:
            norm_val_cm = np.vstack((norm_val_cm, norm_val_cm_row))

    balanced_accuracy = balanced_accuracy_score(y_targ, y_pred)
    classwise_accuracy = confusion_matrix(y_targ, y_pred, normalize="true").diagonal()

    print(classification_report(y_targ, y_pred))
    print(
        "\n — test_f1_macro: {} \n — test_f1_micro: {} \n — test_precis_macro: {} \n — test_precis_micro: {} "
        "\n — test_recall_macro: {} \n — test_recall_micro: {} \n — test_macd: {} \n — test_accuracy: {}"
        "\n — test_balanced_accuracy: {} \n — class wise accuracy: {}".format(val_f1, val_f1_mi, val_precis,
                                                                              val_precis_mi, val_recall,
                                                                              val_recall_mi,
                                                                              val_macd, accuracy,
                                                                              balanced_accuracy,
                                                                              classwise_accuracy))
    with open(log_file, 'a') as f:
        f.writelines(
            "\n — test_f1_macro: {} \n — test_f1_micro: {} \n — test_precis_macro: {} \n — test_precis_micro: {} \n "
            "— test_recall_macro: {} \n — test_recall_micro: {} \n — test_macd: {} \n — test_accuracy: {}"
            "\n — test_balanced_accuracy: {} \n — class wise accuracy: {}\n — normal class distance: {}\n "
            "— squared class distance: {}".format(
                val_f1, val_f1_mi, val_precis,
                val_precis_mi, val_recall,
                val_recall_mi,
                val_macd, accuracy,
                balanced_accuracy,
                classwise_accuracy,
                simple_distance,
                squared_distance))

    class_list = np.unique(y_targ)
    class_names_gt = ['Class_' + str(int(x)) for x in class_list]
    diff_classes = list(set(class_names).difference(class_names_gt))
    index_list = []
    for each_dif in diff_classes:
        index_list.append(class_names.index(each_dif))
    index_list.sort(reverse=False)
    pred_list = list(np.unique(y_pred))
    gt_list = list(np.unique(y_targ))
    additional_pred = [x for x in pred_list if x not in gt_list]
    rem_index_list = [x for x in index_list if x not in additional_pred]
    for dif_index in rem_index_list:
        cm = []
        norm_cm = []

        dif_class = class_names[dif_index]
        class_names_gt.insert(dif_index, dif_class)
        if val_cm.shape[0] != len(class_names):
            for i in range(0, val_cm.shape[1]):
                val_cm_row = list(val_cm[i])
                val_cm_row.insert(dif_index, 0)
                val_cm_row_ins_arr = np.array(val_cm_row)
                if i == 0:
                    cm = val_cm_row_ins_arr
                else:
                    cm = np.vstack((cm, val_cm_row_ins_arr))
        if val_cm.shape[1] != len(class_names):
            val_cm_trans = cm.transpose()
            for i in range(0, val_cm_trans.shape[0]):
                val_cm_row = list(val_cm_trans[i])
                val_cm_row.insert(dif_index, 0)
                val_cm_row_ins_arr = np.array(val_cm_row)
                if i == 0:
                    cm_trans = val_cm_row_ins_arr
                else:
                    cm_trans = np.vstack((cm_trans, val_cm_row_ins_arr))

        if norm_val_cm.shape[0] != len(class_names):
            for i in range(0, norm_val_cm.shape[1]):
                norm_val_cm_row = list(norm_val_cm[i])
                norm_val_cm_row.insert(dif_index, 0)
                norm_val_cm_row_ins_arr = np.array(norm_val_cm_row)
                if i == 0:
                    norm_cm = norm_val_cm_row_ins_arr
                else:
                    norm_cm = np.vstack((norm_cm, norm_val_cm_row_ins_arr))
        if norm_val_cm.shape[1] != len(class_names):
            norm_val_cm_trans = norm_cm.transpose()
            for i in range(0, val_cm_trans.shape[0]):
                norm_val_cm_row = list(norm_val_cm_trans[i])
                norm_val_cm_row.insert(dif_index, 0)
                norm_val_cm_row_ins_arr = np.array(norm_val_cm_row)
                if i == 0:
                    norm_cm_trans = norm_val_cm_row_ins_arr
                else:
                    norm_cm_trans = np.vstack((norm_cm_trans, norm_val_cm_row_ins_arr))

        val_cm = cm_trans.transpose()
        norm_val_cm = norm_cm_trans.transpose()

    cm_path = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '_cm.png')
    norm_cm_path = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '_norm_cm.png')
    class_wise_acc_path = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '_classwise_accuracy.png')
    class_wise_sample_path = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '_classwise_samples.png')

    figure = plot_confusion_matrix(val_cm, class_names=class_names, cm_path=cm_path)
    figure_nor = plot_normalized_confusion_matrix(norm_val_cm, class_names=class_names, cm_path=norm_cm_path)
    classwise_accuracy = confusion_matrix(y_targ, y_pred, normalize="true").diagonal()
    classwise_accuracy_list = list(classwise_accuracy)
    for each_index in rem_index_list:
        classwise_accuracy_list.insert(each_index, 0)

    # plotting classwise accuracy
    plt.bar(class_names, [x * 100 for x in classwise_accuracy_list], align='center')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=75)
    plt.title('Class Wise Accuracy')
    plt.ylabel('Accuracy')
    plt.savefig(class_wise_acc_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()

    plt.bar(class_names, val_cm.sum(axis=1), align='center')
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=75)
    plt.title('Number of samples in each class')
    plt.ylabel('Sample count')
    plt.savefig(class_wise_sample_path, bbox_inches="tight")
    plt.clf()
    plt.cla()
    plt.close()

    data = {}
    data.update({'GRD_ID': ID_list})
    data.update({'GT_POP_CLASS': y_targ})
    data.update({'PR_POP_CLASS': y_pred})
    df = pd.DataFrame(data)

    csv_path = os.path.join(path, path.split(os.sep)[-1] + '_' + dataset_name + '_evaluation.csv')
    df.to_csv(csv_path, index=False)

    return csv_path


def calc_distance(lbls, preds, square=False):
    """
    calculates the mean absolute class distance
    Args:
        lbls: ground truth labels
        preds: predictions
    """
    if square is False:
        score = np.abs(preds-lbls)
    else:
        score = (preds-lbls)**2
    
    return np.mean(score)


def plot_confusion_matrix(cm, class_names, cm_path):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return figure


def plot_normalized_confusion_matrix(cm, class_names, cm_path):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cm_path, bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return figure
