# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from torch import distributed as torch_dist

from nanotron import distributed as dist
from nanotron.distributed import ProcessGroup
def is_rank_0():
    # if torch.cuda.current_device() == 0:
    if torch.distributed.get_rank() == 0:
        return True


class NoOper(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, handle_dic, h_id):
        #ctx.group = group
        print(f"===Guanhua goes into NoOper fwd")
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        print(f"===Guanhua NoOper bwd")
        handle = ctx.handle_dic[ctx.h_id]
        handle.wait()
        return grad_output, None
        #group = ctx.group
        #handle = dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=group, async_op=True)
        #handle = dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=group)
        #return grad_output, None  # applied fwd only

class _async_col_parallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup], handle_dic, h_id):
        print(f"===Guanhua goes into _async_col_para fwd")
        ctx.group = group
        ctx.handle_dic = handle_dic
        ctx.h_id = h_id
        return tensor
    
    @staticmethod
    def backward(ctx, grad_output):
        print(f"===Guanhua _async_col_parallel bwd")
        handle = dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group, async_op=True)
        #dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        ctx.handle_dic[ctx.h_id] = handle
        return grad_output, None


class DifferentiableIdentity(torch.autograd.Function):
    """All-reduce gradients in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        #print(f'===Guanhua DifferentiableIdentity bwd')
        # if is_rank_0():
        #     import pdb; pdb.set_trace()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=group, async_op=True)
        return grad_output, None #DifferentiableAllReduceSum.apply(grad_output, group), None  # applied fwd only


class DifferentiableAllReduceSum(torch.autograd.Function):
    """All-reduce in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        if group.size() == 1:
            return tensor
        #print(f'===Guanhua DifferentiableAllReduceSum fwd')
        # handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group, async_op=True)
        return tensor
        # return handle

    @staticmethod
    def backward(ctx, grad_output):
        #print(f'===Guanhua DifferentiableAllReduceSum bwd')
        return grad_output, None


class DifferentiableAllGather(torch.autograd.Function):
    """All gather in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: gather along another dimension
        sharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        unsharded_batch_size = sharded_batch_size * group.size()

        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # `tensor` can sometimes not be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L317
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        out = DifferentiableReduceScatterSum.apply(grad_output, group)
        return out, None


class DifferentiableReduceScatterSum(torch.autograd.Function):
    """Reduce scatter in a differentiable fashion"""

    @staticmethod
    def forward(ctx, tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        if group.size() == 1:
            return tensor

        # TODO @thomasw21: shard along another dimension
        unsharded_batch_size, *rest_size = tensor.shape
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()
        assert unsharded_batch_size % group.size() == 0

        # TODO @thomasw21: Collectives seem to require tensors to be contiguous
        # https://cs.github.com/pytorch/pytorch/blob/2b267fa7f28e18ca6ea1de4201d2541a40411457/torch/distributed/nn/functional.py#L305
        tensor = tensor.contiguous()

        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )
        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        return DifferentiableAllGather.apply(grad_output, group), None


# -----------------
# Helper functions.
# -----------------
def no_op(tensor, h_dic, h_id):
    return NoOper.apply(tensor, h_dic, h_id)
#(ctx, tensor, group: Optional[ProcessGroup], handle_dic, h_id)
def async_col_par(tensor,h_dic, h_id, group: Optional[ProcessGroup] = None):
    return _async_col_parallel.apply(tensor, group, h_dic, h_id)

def differentiable_identity(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableIdentity.apply(tensor, group)


def differentiable_all_reduce_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllReduceSum.apply(tensor, group)


def differentiable_all_gather(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableAllGather.apply(tensor, group)


def differentiable_reduce_scatter_sum(tensor, group: Optional[ProcessGroup] = None):
    return DifferentiableReduceScatterSum.apply(tensor, group)
