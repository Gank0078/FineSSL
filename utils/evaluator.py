import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import coo_matrix

class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg, cls_num_list=None, **kwargs):
        self.cfg = cfg
        self.cls_num_list = cls_num_list
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = int(label == pred)
            self._per_class_res[label].append(matches)

        labels = list(self._per_class_res.keys())
        labels.sort()

        cls_accs = []
        for label in labels:
            res = self._per_class_res[label]
            correct = sum(res)
            total = len(res)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        
        accs_string = np.array2string(np.array(cls_accs), precision=2)
        print(f"* class acc: {accs_string}")

        many_idxs = np.array(self.cls_num_list) > 100
        few_idxs = np.array(self.cls_num_list) < 20
        med_idxs = ~(many_idxs | few_idxs)

        many_acc = np.mean(np.array(cls_accs)[many_idxs])
        med_acc = np.mean(np.array(cls_accs)[med_idxs])
        few_acc = np.mean(np.array(cls_accs)[few_idxs])

        mean_acc = np.mean(cls_accs)

        results["many_acc"] = many_acc
        results["med_acc"] = med_acc
        results["few_acc"] = few_acc
        results["mean_acc"] = mean_acc

        print(f"* many: {many_acc:.1f}%  med: {med_acc:.1f}%  few: {few_acc:.1f}%")
        print(f"* average: {mean_acc:.1f}%")

        # compute confusion matrix
        # cmat = confusion_matrix(self._y_true, self._y_pred)
        # cmat = coo_matrix(cmat)
        # save_path = osp.join(self.cfg.output_dir, "cmat.pt")
        # torch.save(cmat, save_path)
        # print(f"Confusion matrix is saved to {save_path}")

        return results


def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res
