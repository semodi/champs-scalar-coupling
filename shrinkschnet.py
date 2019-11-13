import schnetpack as spk
from schnetpack.data import Structure
import torch
from torch import nn
import numpy as np
import schnetpack 


class EdgeUpdate(nn.Module):

    def __init__(self, n_atom_basis, n_spatial_basis):
        super(EdgeUpdate, self).__init__()
        self.dense1 = spk.nn.base.Dense(2 * n_atom_basis + n_spatial_basis,
                                        n_atom_basis + n_spatial_basis,activation = spk.nn.activations.shifted_softplus)
        self.dense2 = spk.nn.base.Dense(n_atom_basis + n_spatial_basis,
                                        n_spatial_basis, activation = spk.nn.activations.shifted_softplus)
        
        self.update_network = nn.Sequential(self.dense1, self.dense2)
        

    def forward(self, x, neighbors, f_ij):

#         # calculate filter
#         W = self.filter_network(f_ij)

#         # apply optional cutoff
#         if self.cutoff_network is not None:
#             C = self.cutoff_network(r_ij)
#             #            print(C)
#             W *= C


        nbh_size = neighbors.size()
        nbh = neighbors.view(-1, nbh_size[1] * nbh_size[2], 1)
        nbh = nbh.expand(-1, -1, x.size(2))

        x_gath = torch.gather(x, 1, nbh)
        x_gath = x_gath.view(nbh_size[0], nbh_size[1], nbh_size[2], -1)
        x_gath0 = torch.unsqueeze(x, 2).expand(-1,-1, nbh_size[2], -1).clone()
        f_ij = torch.cat([f_ij, x_gath, x_gath0], dim = -1)
        f_ij = self.update_network(f_ij)
        
        return f_ij
    
class ShrinkSchNet(spk.representation.SchNet):
    
    def __init__(self, n_atom_basis=128, n_filters=128, n_interactions=1, cutoff=5.0, n_gaussians=25,
                 normalize_filter=False, coupled_interactions=False,
                 return_intermediate=False, max_z=100, interaction_block=spk.representation.SchNetInteraction, 
                 edgeupdate_block = EdgeUpdate, trainable_gaussians=False,
                 distance_expansion=None, shrink_layers = 3, shrink_distances = None):
        
        n_interactions += shrink_layers
        if shrink_distances:
            self.shrink_layers = len(shrink_distances)
        else:
            self.shrink_layers = shrink_layers
        self.shrink_distances = shrink_distances
        
        super(ShrinkSchNet, self).__init__(n_atom_basis, n_filters, n_interactions, cutoff, n_gaussians,
                 normalize_filter, coupled_interactions,
                 return_intermediate, max_z, interaction_block, trainable_gaussians,
                 distance_expansion)

        if edgeupdate_block == None:
            self.edge_update = False
            self.edgeupdates = [None] * n_interactions
        else:
            self.edge_update = True

            if coupled_interactions:
                self.edgeupdates = nn.ModuleList([
                                                      edgeupdate_block(n_atom_basis=n_atom_basis,
                                                                        n_spatial_basis=n_gaussians)
                                                  ] * n_interactions)
            else:
                self.edgeupdates = nn.ModuleList([
                    edgeupdate_block(n_atom_basis=n_atom_basis, n_spatial_basis=n_gaussians)
                    for _ in range(n_interactions)
                ])
     

    def forward(self, inputs):

        atomic_numbers = inputs[Structure.Z]
        positions = inputs[Structure.R]
        cell = inputs[Structure.cell]
        cell_offset = inputs[Structure.cell_offset]
        neighbors = inputs[Structure.neighbors]
        neighbor_mask = inputs[Structure.neighbor_mask]
        atom_mask = inputs[Structure.atom_mask]
        
        # atom embedding
        x = self.embedding(atomic_numbers)

        # spatial features
        r_ij = self.distances(positions, neighbors, cell, cell_offset)
        f_ij = self.distance_expansion(r_ij)
        
        nodiagmask = torch.stack([~torch.eye(atom_mask.size()[1]).byte()]*len(atom_mask))

        atom_mask_mat = \
            torch.einsum('ij,ik -> ijk', atom_mask, atom_mask)[nodiagmask].view(len(atom_mask),atom_mask.size()[1],-1)
        
        zero_mask = torch.zeros_like(r_ij)
        
   
        if self.return_intermediate:
            xs = [x]
     
        
        for eupdate, interaction in zip(self.edgeupdates[:-self.shrink_layers], 
                                        self.interactions[:-self.shrink_layers]):
            
            if self.edge_update:
                f_ij = eupdate(x, neighbors, f_ij)
                f_ij = f_ij * neighbor_mask.unsqueeze(-1).expand(-1,-1,-1, f_ij.size()[-1])
                
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            
            x = x + v

            if self.return_intermediate:
                xs.append(xs)    
        
 
        for eupdate, interaction, co_dist in zip(self.edgeupdates[-self.shrink_layers:],
                                                 self.interactions[-self.shrink_layers:], 
                                                 self.shrink_distances):
            if self.edge_update:
                f_ij = eupdate(x, neighbors, f_ij)
                f_ij = f_ij * neighbor_mask.unsqueeze(-1).expand(-1,-1,-1, f_ij.size()[-1])
                
            neighbor_mask = torch.where(r_ij - (10 * atom_mask_mat) < co_dist, neighbor_mask, zero_mask)
            
            v = interaction(x, r_ij, neighbors, neighbor_mask, f_ij=f_ij)
            x = x + v

            if self.return_intermediate:
                xs.append(xs)   
            
        if self.return_intermediate:
            return x, xs
        return x[atom_mask.byte()].view(len(atom_mask),2,-1)

class SCReadout(nn.Module):

    def __init__(self, embed_dim = 128, hidden_dim = 128, mlp_layers = 3, mean=None, stddev=None):
        super(SCReadout, self).__init__()
        self.embed_dim = embed_dim
        self.gru = nn.GRU(input_size=embed_dim,
                               hidden_size=hidden_dim,
                               batch_first = True)
        
        self.mlp = spk.nn.blocks.MLP(hidden_dim, 1, n_layers = mlp_layers)
        self.requires_dr = False
        mean = torch.FloatTensor([0.0]) if mean is None else mean
        stddev = torch.FloatTensor([1.0]) if stddev is None else stddev
        self.standardize = spk.nn.base.ScaleShift(mean, stddev)

    def forward(self, inputs):
        X = inputs['representation']
        Xp = self.gru(X)[1]
        Xm = self.gru(X[:,[1,0]])[1]
        X = (Xp + Xm)[0]
        X = self.mlp(X)
        X = self.standardize(X)

        return {'y': X}

        
    
