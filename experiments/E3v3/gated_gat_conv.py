from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, NoneType, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag


class GatedGATConv(MessagePassing):
    """STAGATE-compatible GAT layer with optional multiplicative edge gates."""

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin_src = nn.Parameter(torch.zeros(size=(in_channels, out_channels)))
        nn.init.xavier_normal_(self.lin_src.data, gain=1.414)
        self.lin_dst = self.lin_src

        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))
        nn.init.xavier_normal_(self.att_src.data, gain=1.414)
        nn.init.xavier_normal_(self.att_dst.data, gain=1.414)

        self._alpha = None
        self.attentions = None

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
        return_attention_weights=None,
        attention: bool = True,
        tied_attention=None,
        edge_gate: Optional[Tensor] = None,
    ):
        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            if x.dim() != 2:
                raise ValueError("Static graphs are not supported in GatedGATConv.")
            x_src = x_dst = torch.mm(x, self.lin_src).view(-1, H, C)
        else:
            x_src, x_dst = x
            if x_src.dim() != 2:
                raise ValueError("Static graphs are not supported in GatedGATConv.")
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        if not attention:
            return x[0].mean(dim=1)

        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if edge_gate is not None:
                raise ValueError("edge_gate is only supported when add_self_loops=False.")
            if isinstance(edge_index, Tensor):
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=x, alpha=alpha, edge_gate=edge_gate, size=size)

        alpha = self._alpha
        if alpha is None:
            raise RuntimeError("attention weights were not computed.")
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            if isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_gate: Optional[Tensor],
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        base_alpha = softmax(alpha, index, ptr, size_i)

        if edge_gate is not None:
            if edge_gate.dim() == 1:
                gate = edge_gate.view(-1, 1)
            elif edge_gate.dim() == 2:
                gate = edge_gate
            else:
                raise ValueError("edge_gate must have shape [num_edges] or [num_edges, heads].")
            gated_alpha = base_alpha * gate
            num_targets = int(index.max().item()) + 1 if size_i is None else size_i
            denom = gated_alpha.new_zeros((num_targets, gated_alpha.size(1)))
            denom.index_add_(0, index, gated_alpha)
            base_alpha = gated_alpha / denom[index].clamp_min(1e-12)

        self._alpha = base_alpha
        base_alpha = F.dropout(base_alpha, p=self.dropout, training=self.training)
        return x_j * base_alpha.unsqueeze(-1)

    def __repr__(self):
        return "{}({}, {}, heads={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.heads,
        )
