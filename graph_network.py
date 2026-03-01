from typing import List
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing


def build_mlp(
        input_size: int,
        hidden_layer_sizes: List[int],
        output_size: int = None,
        output_activation: nn.Module = nn.Identity,
        activation: nn.Module = nn.ReLU) -> nn.Module:
  layer_sizes = [input_size] + hidden_layer_sizes
  if output_size:
    layer_sizes.append(output_size)
  nlayers = len(layer_sizes) - 1
  act = [activation for i in range(nlayers)]
  act[-1] = output_activation
  mlp = nn.Sequential()
  for i in range(nlayers):
    mlp.add_module("NN-" + str(i), nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    mlp.add_module("Act-" + str(i), act[i]())
  return mlp


class Encoder(nn.Module):
  def __init__(self, nnode_in_features, nnode_out_features, nedge_in_features,
               nedge_out_features, nmlp_layers, mlp_hidden_dim):
    super(Encoder, self).__init__()
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in_features,
                                             [mlp_hidden_dim for _ in range(nmlp_layers)],
                                             nnode_out_features),
                                   nn.LayerNorm(nnode_out_features)])
    self.edge_fn = nn.Sequential(*[build_mlp(nedge_in_features,
                                             [mlp_hidden_dim for _ in range(nmlp_layers)],
                                             nedge_out_features),
                                   nn.LayerNorm(nedge_out_features)])

  def forward(self, x, edge_features):
    return self.node_fn(x), self.edge_fn(edge_features)


class InteractionNetwork(MessagePassing):
  def __init__(self, nnode_in, nnode_out, nedge_in, nedge_out, nmlp_layers, mlp_hidden_dim):
    super(InteractionNetwork, self).__init__(aggr='add')
    self.node_fn = nn.Sequential(*[build_mlp(nnode_in + nedge_out,
                                             [mlp_hidden_dim for _ in range(nmlp_layers)],
                                             nnode_out),
                                   nn.LayerNorm(nnode_out)])
    self.edge_fn = nn.Sequential(*[build_mlp(nnode_in + nnode_in + nedge_in,
                                             [mlp_hidden_dim for _ in range(nmlp_layers)],
                                             nedge_out),
                                   nn.LayerNorm(nedge_out)])

  def forward(self, x, edge_index, edge_features):
    x_residual = x
    edge_features_residual = edge_features
    x, edge_features = self.propagate(edge_index=edge_index, x=x, edge_features=edge_features)
    return x + x_residual, edge_features + edge_features_residual

  def message(self, x_i, x_j, edge_features):
    edge_features = torch.cat([x_i, x_j, edge_features], dim=-1)
    self._edge_features = self.edge_fn(edge_features)
    return self._edge_features

  def update(self, x_updated, x, edge_features):
    x_updated = torch.cat([x_updated, x], dim=-1)
    x_updated = self.node_fn(x_updated)
    return x_updated, self._edge_features


class Processor(MessagePassing):
  def __init__(self, nnode_in, nnode_out, nedge_in, nedge_out,
               nmessage_passing_steps, nmlp_layers, mlp_hidden_dim):
    super(Processor, self).__init__(aggr='max')
    self.gnn_stacks = nn.ModuleList([
        InteractionNetwork(
            nnode_in=nnode_in, nnode_out=nnode_out,
            nedge_in=nedge_in, nedge_out=nedge_out,
            nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim,
        ) for _ in range(nmessage_passing_steps)])

  def forward(self, x, edge_index, edge_features):
    for gnn in self.gnn_stacks:
      x, edge_features = checkpoint(gnn, x, edge_index, edge_features, use_reentrant=False)
    return x, edge_features


class Decoder(nn.Module):
  def __init__(self, nnode_in, nnode_out, nmlp_layers, mlp_hidden_dim):
    super(Decoder, self).__init__()
    self.node_fn = build_mlp(nnode_in, [mlp_hidden_dim for _ in range(nmlp_layers)], nnode_out)

  def forward(self, x):
    return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
  def __init__(self, nnode_in_features, nnode_out_features, nedge_in_features,
               latent_dim, nmessage_passing_steps, nmlp_layers, mlp_hidden_dim):
    super(EncodeProcessDecode, self).__init__()
    self._encoder = Encoder(
        nnode_in_features=nnode_in_features, nnode_out_features=latent_dim,
        nedge_in_features=nedge_in_features, nedge_out_features=latent_dim,
        nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim)
    self._processor = Processor(
        nnode_in=latent_dim, nnode_out=latent_dim,
        nedge_in=latent_dim, nedge_out=latent_dim,
        nmessage_passing_steps=nmessage_passing_steps,
        nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim)
    self._decoder = Decoder(
        nnode_in=latent_dim, nnode_out=nnode_out_features,
        nmlp_layers=nmlp_layers, mlp_hidden_dim=mlp_hidden_dim)

  def forward(self, x, edge_index, edge_features):
    x, edge_features = self._encoder(x, edge_features)
    x, edge_features = self._processor(x, edge_index, edge_features)
    x = self._decoder(x)
    return x
