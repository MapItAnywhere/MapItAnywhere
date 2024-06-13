import torch

from .metrics import PixelAccuracy, MeanObservableIOU, MeanUnobservableIOU, ObservableIOU, UnobservableIOU, mAP

from .loss import EnhancedLoss

from .segmentation_head import SegmentationHead

from . import get_model
from .base import BaseModel
from .bev_projection import CartesianProjection, PolarProjectionDepth
from .schema import ModelConfiguration

class MapPerceptionNet(BaseModel):

    def _init(self, conf: ModelConfiguration):
        self.image_encoder = get_model(
            conf.image_encoder.name
        )(conf.image_encoder.backbone)

        self.decoder = SegmentationHead(
            in_channels=conf.latent_dim, n_classes=conf.num_classes)

        ppm = conf.pixel_per_meter
        self.projection_polar = PolarProjectionDepth(
            conf.z_max,
            ppm,
            conf.scale_range,
            conf.z_min,
        )
        self.projection_bev = CartesianProjection(
            conf.z_max, conf.x_max, ppm, conf.z_min
        )

        self.scale_classifier = torch.nn.Linear(
            conf.latent_dim, conf.num_scale_bins
        )  # l4 - working

        self.num_classes = conf.num_classes

        self.loss_fn = EnhancedLoss(conf.loss)

    def _forward(self, data):
        f_image, camera = self.image_encoder(data)

        scales = self.scale_classifier(
            f_image.moveaxis(1, -1))
        f_polar = self.projection_polar(f_image, scales, camera)

        # Map to the BEV.
        f_bev, valid_bev, _ = self.projection_bev(
            f_polar.float(), None, camera.float()
        )

        output = self.decoder(f_bev[..., :-1])

        probs = torch.nn.functional.sigmoid(output)

        return {
            "output": probs,
            "logits": output,
            "scales": scales,
            "features_image": f_image,
            "features_bev": f_bev,
            "valid_bev": valid_bev.squeeze(1),
        }

    def loss(self, pred, data):
        loss = self.loss_fn(pred, data)
        return loss

    def metrics(self):
        m = {
            "pix_acc": PixelAccuracy(),
            "map": mAP(self.num_classes),
            "miou_observable": MeanObservableIOU(self.num_classes),
            "miou_non_observable": MeanUnobservableIOU(self.num_classes),
        }
        m.update(
            {
                f"IoU_observable_class_{i}": ObservableIOU(i, num_classes=self.num_classes)
                for i in range(self.num_classes)
            }
        )
        m.update(
            {
                f"IoU_non_observable_{i}": UnobservableIOU(i, num_classes=self.num_classes)
                for i in range(self.num_classes)
            }
        )
        return m
