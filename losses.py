from typing import Union, Tuple, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from nest import register


@register
def cross_entropy_loss(
    input: Tensor, 
    target: Tensor,
    weight: Optional[Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    """Cross entropy loss.
    """

    return F.cross_entropy(input, target, weight, size_average, ignore_index, reduce)


@register
def smooth_loss(
    input: Tensor,
    target: Tensor,
    smooth_ratio: float = 0.9,
    weight: Union[None, Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    '''Smooth loss.
    '''

    prob = F.log_softmax(input, dim=1)
    ymask = prob.data.new(prob.size()).zero_()
    ymask = ymask.scatter_(1, target.view(-1,1), 1)
    ymask = smooth_ratio*ymask + (1-smooth_ratio)*(1-ymask)/(len(input[1])-1)
    loss = - (prob*ymask).sum(1).mean()

    return loss


@register
def multi_smooth_loss(
    input: Tuple,
    target: Tensor,
    smooth_ratio: float = 0.9,
    loss_weight: Union[None, Dict]= None,
    weight: Union[None, Tensor] = None,
    size_average: bool = True,
    ignore_index: int = -100,
    reduce: bool = True) -> Tensor:
    '''Multi smooth loss.
    '''
    assert isinstance(input, tuple), 'input is less than 2'

    weight_loss = torch.ones(len(input)).to(input[0].device)
    if loss_weight is not None:
        for item in loss_weight.items():
            weight_loss[int(item[0])] = item[1]

    loss = 0
    for i in range(0, len(input)):
        if i in [1, len(input)-1]:
            prob = F.log_softmax(input[i], dim=1)
            ymask = prob.data.new(prob.size()).zero_()
            ymask = ymask.scatter_(1, target.view(-1,1), 1)
            ymask = smooth_ratio*ymask + (1-smooth_ratio)*(1-ymask)/(len(input[i][1])-1)
            loss_tmp = - weight_loss[i]*((prob*ymask).sum(1).mean())
        else:
            loss_tmp = weight_loss[i]*F.cross_entropy(input[i], target, weight, size_average, ignore_index, reduce)
        loss += loss_tmp

    return loss