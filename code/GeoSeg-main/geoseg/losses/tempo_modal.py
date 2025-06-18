import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import Tensor


class TempoModalLoss(nn.Module):

    def __init__(self, main_loss, auxiliary_losses=None, auxiliary_loss_weights={"s1":0.5, "s2":0.5, "planet":0.5}):
        super().__init__()
        self.main_loss = main_loss#nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index,weight=class_weights)
        self.aux_losses = auxiliary_losses
        # self.aux_losses = {
        #     "s1":nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index, weight=class_weights),
        #     "s2":nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index, weight=class_weights),
        #     "planet":nn.CrossEntropyLoss(label_smoothing=smooth_factor, ignore_index=ignore_index, weight=class_weights)
        # }
        self.auxiliary_loss_weights = auxiliary_loss_weights

    def forward(self, forward_dict, labels):
        logit_main = forward_dict["logits"]
        loss = self.main_loss(logit_main, labels)
        loss_out = {
            "main":loss.item(),
            "all":loss,
        }
        if self.aux_losses:
            # TODO solve & discuss
            for modality in forward_dict["non_empty_modalities"]:#forward_dict["non_empty_modalities"][0]:
                logits_aux = forward_dict[f"{modality}_aux"]
                aux_loss = self.auxiliary_loss_weights[modality]*self.aux_losses[modality](logits_aux, labels)
                loss_out[modality] = aux_loss
                loss_out["all"] += aux_loss
        return loss_out


# if __name__ == '__main__':
#     targets = torch.randint(low=0, high=2, size=(2, 16, 16))
#     logits = torch.randn((2, 2, 16, 16))
#     # print(targets)
#     model = EdgeLoss()
#     loss = model.compute_edge_loss(logits, targets)

#     print(loss)