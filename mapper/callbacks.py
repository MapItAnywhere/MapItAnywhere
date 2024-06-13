import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Any
import torchvision
import wandb


class EvalSaveCallback(pl.Callback):

    def __init__(self, save_dir: Path) -> None:
        super().__init__()
        self.save_dir = save_dir

    def save(self, outputs, batch, batch_idx):
        name = batch['name']

        filename = self.save_dir / f"{batch_idx:06d}_{name[0]}.pt"
        torch.save({
            "fpv": batch['image'],
            "seg_masks": batch['seg_masks'],
            'name': name,
            "output": outputs["output"],
            "valid_bev": outputs["valid_bev"],
        }, filename)

    def on_test_batch_end(self, trainer: pl.Trainer,
                          pl_module: pl.LightningModule,
                          outputs: torch.Tensor | Any | None,
                          batch: Any,
                          batch_idx: int,
                          dataloader_idx: int = 0) -> None:
        if not outputs:
            return

        self.save(outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer: pl.Trainer,
                                pl_module: pl.LightningModule,
                                outputs: torch.Tensor | Any | None,
                                batch: Any,
                                batch_idx: int,
                                dataloader_idx: int = 0) -> None:
        if not outputs:

            return

        self.save(outputs, batch, batch_idx)


class ImageLoggerCallback(pl.Callback):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def log_image(self, trainer, pl_module, outputs, batch, batch_idx, mode="train"):
        fpv_rgb = batch["image"]
        fpv_grid = torchvision.utils.make_grid(
            fpv_rgb, nrow=8, normalize=False)
        images = [
            wandb.Image(fpv_grid, caption="fpv")
        ]

        pred = outputs['output'].permute(0, 2, 3, 1)
        pred[outputs["valid_bev"][..., :-1] == 0] = 0
        pred = (pred > 0.5).float()
        pred = pred.permute(0, 3, 1, 2)

        for i in range(self.num_classes):
            gt_class_i = batch['seg_masks'][..., i]
            gt_class_i_grid = torchvision.utils.make_grid(
                gt_class_i.unsqueeze(1), nrow=8, normalize=False, pad_value=0)
            pred_class_i = pred[:, i]
            pred_class_i_grid = torchvision.utils.make_grid(
                pred_class_i.unsqueeze(1), nrow=8, normalize=False, pad_value=0)

            images += [
                wandb.Image(gt_class_i_grid, caption=f"gt_class_{i}"),
                wandb.Image(pred_class_i_grid, caption=f"pred_class_{i}")
            ]

        trainer.logger.experiment.log(
            {
                "{}/images".format(mode): images
            }
        )

    def on_validation_batch_end(self, trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if batch_idx == 0:
            with torch.no_grad():
                outputs = pl_module(batch)
            self.log_image(trainer, pl_module, outputs,
                           batch, batch_idx, mode="val")

    def on_train_batch_end(self, trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        if batch_idx == 0:
            pl_module.eval()

            with torch.no_grad():
                outputs = pl_module(batch)

            self.log_image(trainer, pl_module, outputs,
                           batch, batch_idx, mode="train")

            pl_module.train()
