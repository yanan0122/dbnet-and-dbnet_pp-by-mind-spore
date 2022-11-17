import mindspore.nn as nn
import modules.detector as detector
from modules.backbone import get_backbone


def get_dbnet(net, config, isTrain=True):
    if net == "DBnet":
        return DBnet(config, isTrain)
    if net == "DBnetPP":
        return DBnetPP(config, isTrain)
    raise ValueError(f"Not support net {net}, net should be in [DBnet, DBnetPP]")


class DBnet(nn.Cell):

    def __init__(self, config, isTrain=True):
        super(DBnet, self).__init__(auto_prefix=False)

        self.backbone = get_backbone(config['backbone']['initializer'])(config['backbone']['pretrained'],
                                                                        config['backbone']['pretrained_ckpt'])
        self.segdetector = detector.SegDetector(training=isTrain, **config['segdetector'])

    def construct(self, img):
        pred = self.backbone(img)
        pred = self.segdetector(pred)

        return pred


class DBnetPP(nn.Cell):
    def __init__(self, config, isTrain=True):
        super(DBnetPP, self).__init__(auto_prefix=False)

        self.backbone = get_backbone(config['backbone']['initializer'])(config['backbone']['pretrained'],
                                                                        config['backbone']['pretrained_ckpt'])
        self.segdetector = detector.SegDetectorPP(training=isTrain, **config['segdetector'])

    def construct(self, img):
        pred = self.backbone(img)
        pred = self.segdetector(pred)
        return pred


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)

        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, img, gt, gt_mask, thresh_map, thresh_mask):
        pred = self._backbone(img)
        loss = self._loss_fn(pred, gt, gt_mask, thresh_map, thresh_mask)

        return loss

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone