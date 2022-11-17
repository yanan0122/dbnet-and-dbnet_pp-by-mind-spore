import sys
import time

import numpy as np
from mindspore import Tensor, nn, ops
import mindspore as ms
import mindspore.numpy as mnp

# Input:
#             pred: A dict which contains predictions.
#                 thresh: The threshold prediction
#                 binary: The text segmentation prediction.
#                 thresh_binary: Value produced by `step_function(binary - thresh)`.
#             batch:
#                 gt: Text regions bitmap gt.
#                 mask: Ignore mask,
#                     pexels where value is 1 indicates no contribution to loss.
#                 thresh_mask: Mask indicates regions cared by thresh supervision.
#                 thresh_map: Threshold gt.

class L1BalanceCELoss(nn.LossBase):
    '''
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    '''

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, bce_replace="bceloss"):
        super(L1BalanceCELoss, self).__init__()

        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        
        if bce_replace == "bceloss":
            self.bce_loss = BalanceCrossEntropyLoss()
        elif bce_replace == "focalloss":
            self.bce_loss = myFocalLoss()
        elif bce_replace == "diceloss":
            self.bce_loss = DiceLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def construct(self, pred, gt, gt_mask, thresh_map, thresh_mask):
        bce_loss_output = self.bce_loss(pred['binary'], gt, gt_mask)   

        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], thresh_map, thresh_mask)
            dice_loss = self.dice_loss(pred['thresh_binary'], gt, gt_mask)
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss_output * self.bce_scale
        else:
            loss = bce_loss_output

        return loss


class myFocalLoss(nn.LossBase):
    
    def __init__(self):
        super(myFocalLoss,self).__init__()
        self.Focalloss = nn.FocalLoss()
        
    def construct(self, pred, gt, mask):
        
        pred = pred * mask
        gt = gt * mask
        loss = self.Focalloss(pred, gt)
        return loss
        
        

class DiceLoss(nn.LossBase):

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def construct(self, pred: Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of two heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)
        if weights is not None:
            mask = weights * mask

        intersection = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union

        return loss


class MaskL1Loss(nn.LossBase):

    def __init__(self, eps=1e-6):

        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def construct(self, pred, gt, mask):
        pred = pred.squeeze(axis=1)
        mask_sum = mask.sum()
        return ((pred - gt).abs() * mask).sum() / (mask_sum + self.eps)


class BalanceCrossEntropyLoss(nn.LossBase):
    '''
    Balanced cross entropy loss.
    Shape:
        - Input: :math:`(N, 1, H, W)`
        - GT: :math:`(N, 1, H, W)`, same shape as the input
        - Mask: :math:`(N, H, W)`, same spatial shape as the input
        - Output: scalar.

    '''

    def __init__(self, negative_ratio=3, eps=1e-6):

        super(BalanceCrossEntropyLoss, self).__init__()

        self.negative_ratio = negative_ratio
        self.eps = eps
        self.bceloss = nn.BCELoss(reduction="none")
        # self.sort = ops.TopK()
        self.sort = ops.Sort(descending=False)
        self.min = ops.Minimum()
        self.cast = ops.Cast()
        self.gather = ops.GatherNd()
        self.stack = ops.Stack(axis=1)
        self.unsqueeze = ops.ExpandDims()

    def construct(self, pred, gt, mask):

        '''
        Args:
            pred: shape :math:`(N, 1, H, W)`, the prediction of network
            gt: shape :math:`(N, 1, H, W)`, the target
            mask: shape :math:`(N, H, W)`, the mask indicates positive regions
        '''

        # see this example for workaround of hard negative mining:
        # https://gitee.com/zhao_ting_v/ssd_benchmark/blob/master/src/ssd_benchmark.py
        pred = pred.squeeze(axis=1)
        gt = gt.squeeze(axis=1)
        pos = (gt * mask).astype(ms.float32)
        neg = ((1 - gt) * mask).astype(ms.float32)

        positive_count = pos.sum(axis=(1, 2), keepdims=True).astype(ms.int32)
        negative_count = neg.sum(axis=(1, 2), keepdims=True).astype(ms.int32)

        negative_count = self.min(negative_count, positive_count * self.negative_ratio).squeeze(axis=(1, 2))

        loss = self.bceloss(pred.astype(ms.float32), gt.astype(ms.float32))

        positive_loss = loss * pos
        N = loss.shape[0]
        negative_loss = (loss * neg).view(N, -1)

        # negative_value, _ = self.sort(negative_loss, negative_loss.shape[1]) # sort the losses in descending order.
        negative_value, _ = self.sort(negative_loss)
        batch_iter = mnp.arange(N)
        neg_index = self.stack((batch_iter, negative_count))
        min_neg_score = self.unsqueeze(self.gather(negative_value, neg_index), 1)

        masked_neg_loss = self.cast(negative_loss >= min_neg_score, ms.float32) # filter out losses less than topk loss.

        masked_neg_loss = ops.stop_gradient(masked_neg_loss)

        masked_neg_loss = masked_neg_loss * negative_loss

        balance_loss = (positive_loss.sum() + masked_neg_loss.sum()) / \
                       ((positive_count + negative_count).astype(ms.float32).sum() + self.eps)

        return balance_loss

def test_bce():
    # pred = np.load("/old/wlh/DBnetpp_mindspore/test_np/pred.npy")
    # pred = Tensor(pred)
    # mask = np.load("/old/wlh/DBnetpp_mindspore/test_np/mask.npy")
    # mask = Tensor(mask)
    # gt = np.load("/old/wlh/DBnetpp_mindspore/test_np/gt.npy")
    # gt = Tensor(gt)
    # print(pred.shape, mask.shape, gt.shape, pred.dtype, mask.dtype, gt.dtype)
    # sys.exit()

    BCE = BalanceCrossEntropyLoss()

    SHAPE = 640

    for _ in range(20):

        pred_random = Tensor(np.random.rand(16,1,SHAPE,SHAPE), dtype=ms.float32)
        gt_random = Tensor(np.random.rand(16,1,SHAPE,SHAPE), dtype=ms.float32)
        mask_random = Tensor(np.random.rand(16,SHAPE,SHAPE), dtype=ms.float32)

        start = time.time()
        loss = BCE(pred_random, gt_random, mask_random)
        end = time.time()

        print("loss ",loss)
        print("time ", end-start)

def compare_loss():
    SHAPE = 640
    np.random.seed(1)
    pred_random = Tensor(np.random.rand(1,SHAPE,SHAPE), dtype=ms.float32)
    np.random.seed(2)
    gt_random = Tensor(np.random.rand(1,SHAPE,SHAPE), dtype=ms.float32)
    np.random.seed(3)
    mask_random = Tensor(np.random.rand(1,SHAPE,SHAPE), dtype=ms.float32)
    
    
    # focal = myFocalLoss()
    # loss = focal(pred_random, gt_random, mask_random)
    # print("focalloss:{}".format(loss))
    
    dice = DiceLoss()
    loss1 = dice(pred_random, gt_random, mask_random)
    print("diceloss:{}".format(loss1))

    maskl1 = MaskL1Loss()
    loss2 = maskl1(pred_random, gt_random, mask_random)
    print(f"maskl1loss:{loss2}")
    
    
    

if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.PYNATIVE_MODE, device_id=3)
    test_bce()

# update sort():
# "/usr/local/Ascend/ascend-toolkit/5.0.4/arm64-linux/opp/op_impl/built-in/ai_core/tbe/impl/sort.py"