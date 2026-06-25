from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, NoneType, OptPairTensor, OptTensor, Size
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag


class GatedGATConv(MessagePassing):
    """A baseline-faithful copy of STAGATE's GATConv with optional edge gates.

    The official ``STAGATE_pyG.gat_conv.GATConv`` is intentionally not modified.
    This layer keeps the same parameterization and attention computation, then
    multiplies normalized attention by a continuous edge gate during message
    passing.
    """

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)
        del bias

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
        edge_gate: OptTensor = None,
        size: Size = None,
        return_attention_weights=None,
        attention: bool = True,
        tied_attention=None,
    ):
        # type: ignore[no-untyped-def]
        heads, out_channels = self.heads, self.out_channels

        if isinstance(x, Tensor):
            if x.dim() != 2:
                raise ValueError("Static graphs are not supported in GatedGATConv")
            x_src = x_dst = torch.mm(x, self.lin_src).view(
                -1,
                heads,
                out_channels,
            )
        else:
            x_src, x_dst = x
            if x_src.dim() != 2:
                raise ValueError("Static graphs are not supported in GatedGATConv")
            x_src = self.lin_src(x_src).view(-1, heads, out_channels)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, heads, out_channels)

        x_pair = (x_src, x_dst)

        if not attention:
            return x_pair[0].mean(dim=1)

        if tied_attention is None:
            alpha_src = (x_src * self.att_src).sum(dim=-1)
            alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
            alpha = (alpha_src, alpha_dst)
            self.attentions = alpha
        else:
            alpha = tied_attention

        if self.add_self_loops:
            if not isinstance(edge_index, Tensor):
                edge_index = set_diag(edge_index)
            else:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
                if edge_gate is not None:
                    raise ValueError(
                        "edge_gate alignment is only supported when "
                        "add_self_loops=False"
                    )

        out = self.propagate(
            edge_index,
            x=x_pair,
            alpha=alpha,
            edge_gate=edge_gate,
            size=size,
        )

        alpha_out = self._alpha
        if alpha_out is None:
            raise RuntimeError("Attention weights were not populated")
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha_out)
            if isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha_out, layout="coo")
        return out

    def message(
        self,
        x_j: Tensor,
        alpha_j: Tensor,
        alpha_i: OptTensor,
        edge_gate: OptTensor,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = torch.sigmoid(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if edge_gate is not None:
            alpha = alpha * edge_gate.view(-1, 1)
        return x_j * alpha.unsqueeze(-1)


class EdgeGate(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        gate_dim: int,
        *,
        use_distribution: bool,
        use_consistency: bool,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(embedding_dim, gate_dim, bias=False)
        self.key = nn.Linear(embedding_dim, gate_dim, bias=False)
        self.raw_beta = nn.Parameter(torch.tensor(0.0))
        self.raw_eta = nn.Parameter(torch.tensor(0.0))
        self.use_distribution = use_distribution
        self.use_consistency = use_consistency
        self.scale = gate_dim**0.5

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.raw_beta)

    @property
    def eta(self) -> Tensor:
        return F.softplus(self.raw_eta)

    def _directed_logits(
        self,
        warmup_embedding: Tensor,
        source: Tensor,
        target: Tensor,
        distribution_z: Tensor,
        log_consistency: Tensor,
    ) -> Tensor:
        query = self.query(warmup_embedding)
        key = self.key(warmup_embedding)
        logits = (query[source] * key[target]).sum(dim=1) / self.scale
        if self.use_distribution:
            logits = logits - self.beta * distribution_z
        if self.use_consistency:
            logits = logits + self.eta * log_consistency
        return logits

    def edge_gates(
        self,
        warmup_embedding: Tensor,
        edge_index: Tensor,
        edge_pair_id: Tensor,
        edge_is_self_loop: Tensor,
        pair_distribution_z: Tensor,
        pair_log_consistency: Tensor,
    ) -> Tensor:
        source = edge_index[0]
        target = edge_index[1]
        gates = torch.ones(
            edge_index.shape[1],
            dtype=warmup_embedding.dtype,
            device=warmup_embedding.device,
        )
        non_self = ~edge_is_self_loop
        pair_id = edge_pair_id[non_self]
        logits = self._directed_logits(
            warmup_embedding,
            source[non_self],
            target[non_self],
            pair_distribution_z[pair_id],
            pair_log_consistency[pair_id],
        )
        gates[non_self] = torch.sigmoid(logits)
        return gates

    def pair_gates(
        self,
        warmup_embedding: Tensor,
        pair_node_a: Tensor,
        pair_node_b: Tensor,
        pair_distribution_z: Tensor,
        pair_log_consistency: Tensor,
    ) -> Tensor:
        logits_ab = self._directed_logits(
            warmup_embedding,
            pair_node_a,
            pair_node_b,
            pair_distribution_z,
            pair_log_consistency,
        )
        logits_ba = self._directed_logits(
            warmup_embedding,
            pair_node_b,
            pair_node_a,
            pair_distribution_z,
            pair_log_consistency,
        )
        return 0.5 * (torch.sigmoid(logits_ab) + torch.sigmoid(logits_ba))


class GatedSTAGATE(nn.Module):
    def __init__(self, hidden_dims: list[int]) -> None:
        super().__init__()
        in_dim, num_hidden, out_dim = hidden_dims
        self.conv1 = GatedGATConv(
            in_dim,
            num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv2 = GatedGATConv(
            num_hidden,
            out_dim,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv3 = GatedGATConv(
            out_dim,
            num_hidden,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )
        self.conv4 = GatedGATConv(
            num_hidden,
            in_dim,
            heads=1,
            concat=False,
            dropout=0,
            add_self_loops=False,
            bias=False,
        )

    def forward(
        self,
        features: Tensor,
        edge_index: Tensor,
        edge_gate: OptTensor = None,
    ) -> Tuple[Tensor, Tensor]:
        h1 = F.elu(self.conv1(features, edge_index, edge_gate=edge_gate))
        h2 = self.conv2(h1, edge_index, attention=False)
        self.conv3.lin_src.data = self.conv2.lin_src.transpose(0, 1)
        self.conv3.lin_dst.data = self.conv2.lin_dst.transpose(0, 1)
        self.conv4.lin_src.data = self.conv1.lin_src.transpose(0, 1)
        self.conv4.lin_dst.data = self.conv1.lin_dst.transpose(0, 1)
        h3 = F.elu(
            self.conv3(
                h2,
                edge_index,
                edge_gate=edge_gate,
                attention=True,
                tied_attention=self.conv1.attentions,
            )
        )
        h4 = self.conv4(h3, edge_index, attention=False)
        return h2, h4
