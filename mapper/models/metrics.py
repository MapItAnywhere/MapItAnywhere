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
        self.add_state("intersection_observable", default=torch.zeros(
            num_classes), dist_reduce_fx="sum")
        self.add_state("union_observable", default=torch.zeros(
            num_classes), dist_reduce_fx="sum")
        self.add_state("intersection_non_observable",
                       default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("union_non_observable", default=torch.zeros(
            num_classes), dist_reduce_fx="sum")

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
            intersection_observable = torch.logical_and(
                torch.logical_and(pred_mask, gt_mask), observable_mask
            ).sum()
            union_observable = torch.logical_and(
                torch.logical_or(pred_mask, gt_mask), observable_mask
            ).sum()
            self.intersection_observable[class_idx] += intersection_observable
            self.union_observable[class_idx] += union_observable

            # For non-observable areas
            intersection_non_observable = torch.logical_and(
                torch.logical_and(pred_mask, gt_mask), non_observable_mask
            ).sum()
            union_non_observable = torch.logical_and(
                torch.logical_or(pred_mask, gt_mask), non_observable_mask
            ).sum()

            self.intersection_non_observable[class_idx] += intersection_non_observable
            self.union_non_observable[class_idx] += union_non_observable

    def compute(self):
        raise NotImplemented


class ObservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        return (self.intersection_observable / (self.union_observable + 1e-6))[self.class_idx]


class UnobservableIOU(IOU):
    def __init__(self, class_idx=0, **kwargs):
        super().__init__(**kwargs)
        self.class_idx = class_idx

    def compute(self):
        return (self.intersection_non_observable / (self.union_non_observable + 1e-6))[self.class_idx]


class MeanObservableIOU(IOU):
    def compute(self):
        return self.intersection_observable.sum() / (self.union_observable.sum() + 1e-6)


class MeanUnobservableIOU(IOU):
    def compute(self):
        return self.intersection_non_observable.sum() / (self.union_non_observable.sum() + 1e-6)


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
