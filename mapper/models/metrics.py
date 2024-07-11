import torch
import torchmetrics
import torchmetrics.classification


class PixelAccuracy(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct_pixels", default=torch.tensor(
            0), dist_reduce_fx="sum")
        self.add_state("total_pixels", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def update(self, pred, data):
        output_mask = pred['output'] > 0.5
        gt_mask = data["seg_masks"].permute(0, 3, 1, 2)
        self.correct_pixels += (
            (output_mask == gt_mask).sum()
        )
        self.total_pixels += torch.numel(pred["valid_bev"][..., :-1])

    def compute(self):
        return self.correct_pixels / self.total_pixels


class IOU(torchmetrics.Metric):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.add_state("tp_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("tp_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fp_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn_non_observable", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, output, data):

        gt = data["seg_masks"]
        pred = output['output']

        if "confidence_map" in data:
            observable_mask = torch.logical_and(
                output["valid_bev"][..., :-1], data["confidence_map"] == 0)
            non_observable_mask = torch.logical_and(
                output["valid_bev"][..., :-1], data["confidence_map"] == 1)
        else:
            observable_mask = output["valid_bev"][..., :-1]
            non_observable_mask = torch.logical_not(observable_mask)

        for class_idx in range(self.num_classes):
            pred_mask = pred[:, class_idx] > 0.5
            gt_mask = gt[..., class_idx]

            # For observable areas
            gt_mask_bool = gt_mask.bool()
            tp_observable = torch.logical_and(
                torch.logical_and(pred_mask, gt_mask_bool), observable_mask
            ).sum()
            fn_observable = torch.logical_and(
                torch.logical_and(gt_mask_bool, ~pred_mask), observable_mask
            ).sum()
            fp_observable = torch.logical_and(
                torch.logical_and(~gt_mask_bool, pred_mask), observable_mask
            ).sum()
            
            # For non-observable areas
            tp_non_observable = torch.logical_and(
                torch.logical_and(pred_mask, gt_mask_bool), non_observable_mask
            ).sum()
            fn_non_observable = torch.logical_and(
                torch.logical_and(gt_mask_bool, ~pred_mask), non_observable_mask
            ).sum()
            fp_non_observable = torch.logical_and(
                torch.logical_and(~gt_mask_bool, pred_mask), non_observable_mask
            ).sum()
            
            # Update the state
            self.tp_observable[class_idx] += tp_observable
            self.fn_observable[class_idx] += fn_observable
            self.fp_observable[class_idx] += fp_observable
            self.tp_non_observable[class_idx] += tp_non_observable
            self.fn_non_observable[class_idx] += fn_non_observable
            self.fp_non_observable[class_idx] += fp_non_observable

    def compute(self):
        raise NotImplemented


class ObservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        # return (self.intersection_observable / (self.union_observable + 1e-6))[self.class_idx]
        intersection_observable = self.tp_observable[self.class_idx]
        union_observable = self.tp_observable[self.class_idx] + self.fn_observable[self.class_idx] + self.fp_observable[self.class_idx]
        return intersection_observable / (union_observable + 1e-6)
    
class UnobservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        # return (self.intersection_non_observable / (self.union_non_observable + 1e-6))[self.class_idx]
        intersection_non_observable = self.tp_non_observable[self.class_idx]
        union_non_observable = self.tp_non_observable[self.class_idx] + self.fn_non_observable[self.class_idx] + self.fp_non_observable[self.class_idx]
        return intersection_non_observable / (union_non_observable + 1e-6)
    
class MeanObservableIOU(IOU):
    def compute(self):
        # return self.intersection_observable.sum() / (self.union_observable.sum() + 1e-6)
        intersection_observable = self.tp_observable.sum()
        union_observable = self.tp_observable.sum() + self.fn_observable.sum() + self.fp_observable.sum()
        return intersection_observable / (union_observable + 1e-6)

class MeanUnobservableIOU(IOU):
    def compute(self):
        # return self.intersection_non_observable.sum() / (self.union_non_observable.sum() + 1e-6)
        intersection_non_observable = self.tp_non_observable.sum()
        union_non_observable = self.tp_non_observable.sum() + self.fn_non_observable.sum() + self.fp_non_observable.sum()
        return intersection_non_observable / (union_non_observable + 1e-6)
    
class mAP(torchmetrics.classification.MultilabelPrecision):
    def __init__(self, num_labels, **kwargs):
        super().__init__(num_labels=num_labels, **kwargs)

    def update(self, output, data):

        if "confidence_map" in data:
            observable_mask = torch.logical_and(
                output["valid_bev"][..., :-1], data["confidence_map"] == 0)
        else:
            observable_mask = output["valid_bev"][..., :-1]

        pred = output['output']
        pred = pred.permute(0, 2, 3, 1)
        pred = pred[observable_mask]

        target = data['seg_masks']
        target = target[observable_mask]

        super(mAP, self).update(pred, target)
