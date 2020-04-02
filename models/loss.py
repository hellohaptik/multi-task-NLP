
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
from enum import IntEnum

class LossType(IntEnum):
    CrossEntropyLoss = 0
    SpanLoss = 1

class CrossEntropyLoss(_Loss):
    def __init__(self, alpha=1.0, name='Cross Entropy Loss'):
        super().__init__()
        
        self.alpha = alpha
        self.name = name

    def forward(self, inp, target, ignore_index = -1):
        loss = F.cross_entropy(input, target, ignore_index=ignore_index) 
        loss *= self.alpha
        return loss

class SpanLoss(_Loss):
    def __init__(self, alpha=1.0, name='Span Cross Entropy Loss'):
        super().__init__()

        self.alpha = alpha
        self.name = name
    def forward(self, inp, target, ignore_index = -1):

        #assert if inp and target has both start and end values
        assert len(inp) == 2, "start and end logits should be present for span loss calc"
        assert len(target) == 2, "start and end logits should be present for span loss calc"

        startInp, endInp = inp
        startTarg, endTarg = target
        
        startloss = F.cross_entropy(startInp, startTarg, ignore_index=ignore_index)
        endLoss = F.cross_entropy(endInp, endTarg, ignore_index=ignore_index)

        loss = 0.5 * (startloss + endLoss) * self.alpha
        return loss


