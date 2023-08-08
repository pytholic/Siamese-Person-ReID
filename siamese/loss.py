import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet loss
    Input:
        anchor embedding, positive embedding and negative embedding
    """

    def __init__(self, margin, soft):
        super().__init__()
        self.margin = margin
        self.soft = soft

    def forward(
        self, anchor, positive, negative, soft=False, size_average=True
    ):
        similarity_positive = F.cosine_similarity(anchor, positive)
        similarity_negative = F.cosine_similarity(anchor, negative)
        if self.soft:
            loss = torch.log(
                1 + torch.exp(similarity_positive - similarity_negative)
            )
        else:
            #loss = (1- similarity_positive)**2 + (0 - similarity_negative)**2
            loss = torch.relu(
                similarity_negative - similarity_positive + self.margin
            )
        return loss.mean() if size_average else loss.sum()
