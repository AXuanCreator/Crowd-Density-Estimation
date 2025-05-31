import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class P2P_Loss(nn.Module):
    def __init__(self):
        """Create the criterion.
        Parameters:
                num_classes: number of object categories, omitting the special no-object category
                matcher: module able to compute a matching between targets and proposals
                weight_dict: dict containing as key the names of the losses and as values their relative weight.
                eos_coef: relative classification weight applied to the no-object category
                losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = 1
        self.matcher = HungarianMatcher_Crowd(cost_class=1, cost_point=0.05)
        self.weight_dict = {"loss_ce": 1, "loss_points": 0.0002}
        self.eos_coef = 0.5
        self.losses = ["labels", "points"]

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """
        对分类结果进行损失计算——CrossEntropy
        Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )  # gt
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        return losses

    def loss_points(self, outputs, targets, indices, num_points):
        """
        对点位置进行损失计算——MSE
        Args:
                outputs:
                targets:
                indices:
                num_points:

        Returns:

        """
        assert "pred_points" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        loss_bbox = F.mse_loss(src_points, target_points, reduction="none")

        losses = {}
        losses["loss_point"] = loss_bbox.sum() / num_points

        return losses

    def _get_src_permutation_idx(self, indices):
        """

        Args:
                indices:

        Returns:

        """
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "points": self.loss_points,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_points, **kwargs)

    def _is_dist_avail_and_initialized(self):
        if not dist.is_available():
            return False
        if not dist.is_initialized():
            return False
        return True

    def _get_world_size(self):
        if not self._is_dist_avail_and_initialized():
            return 1
        return dist.get_world_size()

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
                 outputs: dict of tensors, see the output specification of the model for the format
                 targets: list of dicts, such that len(targets) == batch_size.
                                  The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
        }

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=next(iter(output1.values())).device
        )
        if self._is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / self._get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, output1, targets, indices1, num_boxes))

        return losses


class HungarianMatcher_Crowd(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
    """

    def __init__(self, cost_class: float = 1, cost_point: float = 1):
        """Creates the matcher

        Params:
                cost_class: This is the relative weight of the foreground object
                cost_point: This is the relative weight of the L1 error of the points coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_point = cost_point
        assert cost_class != 0 or cost_point != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
                outputs: This is a dict that contains at least these entries:
                         "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                         "points": Tensor of dim [batch_size, num_queries, 2] with the predicted point coordinates

                targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                         "labels": Tensor of dim [num_target_points] (where num_target_points is the number of ground-truth
                                           objects in the target) containing the class labels
                         "points": Tensor of dim [num_target_points, 2] containing the target point coordinates

        Returns:
                A list of size batch_size, containing tuples of (index_i, index_j) where:
                        - index_i is the indices of the selected predictions (in order)
                        - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                        len(index_i) = len(index_j) = min(num_queries, num_target_points)
        """
        # 获取batch_size和检测点数量
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 将分类对数和点坐标展平(batch_size合并)，分类对数将经过Softmax处理
        out_prob = (
            outputs["pred_logits"].flatten(0, 1).softmax(-1)
        )  # [batch_size * num_queries, num_classes]
        out_points = outputs["pred_points"].flatten(
            0, 1
        )  # [batch_size * num_queries, 2]

        # 拼接目标标签和点坐标，使其与out_prob和out_points相对应
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_points = torch.cat([v["point"] for v in targets])

        # 计算分类成本，使用负的概率值
        cost_class = -out_prob[:, tgt_ids]

        # 计算点之间的L2距离成本
        cost_point = torch.cdist(out_points, tgt_points, p=2)

        # 计算最终成本
        C = self.cost_point * cost_point + self.cost_class * cost_class
        C = C.view(bs, num_queries, -1).cpu()

        # 获取每个目标大小
        sizes = [len(v["point"]) for v in targets]
        # 使用先行分配算法匹配每个批次的元素
        indices = [
            linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
