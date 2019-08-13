import math
from typing import Union, Tuple, Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from nest import register, Context


class AverageMeter(object):

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


@register
def multi_topk_meter(
	ctx: Context, 
	train_ctx: Context, 
	k: int=1, 
	init_num: int=1, 
	end_num: int = 0) -> dict:
	"""Multi topk meter.
	"""

	def accuracy(output, target, k=1):
		batch_size = target.size(0)

		_, pred = output.topk(k, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		return correct_k.mul_(100.0 / batch_size)

	for i in range(init_num, len(train_ctx.output) - end_num):
		if not "branch_"+str(i) in ctx:
			setattr(ctx, "branch_"+str(i), AverageMeter())

	if train_ctx.batch_idx == 0:
		for i in range(init_num, len(train_ctx.output) - end_num):
			getattr(ctx, "branch_"+str(i)).reset()

	for i in range(init_num, len(train_ctx.output) - end_num):
		acc = accuracy(train_ctx.output[i], train_ctx.target, k)
		getattr(ctx, "branch_"+str(i)).update(acc.item())

	acc_list = {}

	for i in range(init_num, len(train_ctx.output) - end_num):
		acc_list["branch_"+str(i)] = getattr(ctx, "branch_"+str(i)).avg

	return acc_list


@register
def best_meter(ctx: Context, train_ctx: Context, best_branch: int = 1, k: int = 1) -> float:
	"""Best meter.
	"""
	def accuracy(output, target, k=1):
		batch_size = target.size(0)

		_, pred = output.topk(k, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		return correct_k.mul_(100.0 / batch_size)
	
	if not 'meter' in ctx:
		ctx.meter = AverageMeter()

	if train_ctx.batch_idx == 0:
		ctx.meter.reset()
	acc = accuracy(train_ctx.output[best_branch], train_ctx.target, k)
	ctx.meter.update(acc.item())
	return ctx.meter.avg