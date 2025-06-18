import numpy as np


class Evaluator(object):
    def __init__(self, num_class, ignore_index=-1):
        self.num_class = num_class
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        # Exclude the ignore_index row and column from calculations
        valid_indices = [i for i in range(self.num_class) if i != self.ignore_index]
        valid_cm = self.confusion_matrix[valid_indices][:, valid_indices]
        
        tp = np.diag(valid_cm)
        fp = valid_cm.sum(axis=0) - np.diag(valid_cm)
        fn = valid_cm.sum(axis=1) - np.diag(valid_cm)
        tn = np.sum(valid_cm) - tp - fp - fn
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn )
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp + self.eps)
        Recall = tp / (tp + fn + self.eps)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall)
        return F1

    def OA(self):
        valid_pixels = np.sum(self.confusion_matrix) - np.sum(self.confusion_matrix[self.ignore_index]) - np.sum(self.confusion_matrix[:, self.ignore_index])
        OA = (np.sum(np.diag(self.confusion_matrix)) - self.confusion_matrix[self.ignore_index, self.ignore_index]) / (valid_pixels + self.eps)
        return OA

    def Intersection_over_Union(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / (2 * tp + fp + fn + self.eps)
        return Dice

    def Pixel_Accuracy_Class(self):
        valid_indices = [i for i in range(self.num_class) if i != self.ignore_index]
        valid_cm = self.confusion_matrix[valid_indices][:, valid_indices]
        Acc = np.diag(valid_cm) / (valid_cm.sum(axis=1) + self.eps)
        return Acc

    def Frequency_Weighted_Intersection_over_Union(self):
        valid_indices = [i for i in range(self.num_class) if i != self.ignore_index]
        valid_cm = self.confusion_matrix[valid_indices][:, valid_indices]
        freq = np.sum(valid_cm, axis=1) / (np.sum(valid_cm) + self.eps)
        iou = self.Intersection_over_Union()
        FWIoU = (freq[freq > 0] * iou[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape, gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)



if __name__ == '__main__':

    gt = np.array([[0, 2, 1],
                   [1, 2, 1],
                   [1, 0, 1]])

    pre = np.array([[1, 1, 1],
                   [2, 2, 1],
                   [1, 1, 1]])

    eval = Evaluator(num_class=3, ignore_index=0)
    eval.add_batch(gt, pre)
    print(eval.confusion_matrix)
    print(eval.get_tp_fp_tn_fn())
    print(eval.Precision())
    print(eval.Recall())
    print(eval.Intersection_over_Union())
    print(eval.OA())
    print(eval.F1())
    print(eval.Frequency_Weighted_Intersection_over_Union())
