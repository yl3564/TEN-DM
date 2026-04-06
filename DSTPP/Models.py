import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import DSTPP.Constants as Constants
from DSTPP.Layers import EncoderLayer

from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def get_non_pad_mask(seq):
    """ Get the non-padding positions (1 - nonpadding ; 0 - padding). """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1,2,0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # b x ls x ls
    return subsequent_mask

class Attn_Fuse(nn.Module):
    def __init__(self, d=64, nhead=4, hidden=128, dropout=0.5):
        super().__init__()
        self.reduce = nn.Linear(2*d, d)  # [e_s,e_t] -> d
        self.mha = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(2*d, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, d)
        )
        self.ln = nn.LayerNorm(d)

    def forward(self, e_s, e_t, e_topo, key_padding_mask=None):
        B, L, D = e_s.shape
        e = self.reduce(torch.cat([e_s, e_t], dim=-1)) # (B,L,d)
        Q = e_topo.unsqueeze(1) # (B,1,d)
        g, _ = self.mha(Q, e, e, key_padding_mask=key_padding_mask) # (B,1,d)
        g = g.squeeze(1).unsqueeze(1).expand(B, L, D) # (B,L,d)

        z = self.ln(e + g)
        z = self.proj(torch.cat([z, e], dim=-1))
        return z  # (B,L,d)

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.Modulelist([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_loc)
        
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

def build_graph(event_time, event_loc, connect_ratio):
    graphs = []

    for times, locs in zip(event_time, event_loc):
        times = times.unsqueeze(1)  # (N, 1)
        node_features = torch.cat([times, locs], dim=1)  # (N, 3)
        N = node_features.size(0)

        dists_matrix = torch.cdist(locs, locs)  # (N, N)
        dists_matrix.fill_diagonal_(float('inf'))

        unique_node_pair = torch.triu_indices(N, N, offset=1) # all unique unordered node pairs without repetition
        dist_values = dists_matrix[unique_node_pair[0], unique_node_pair[1]]
        k = int(connect_ratio * len(dist_values))
        top_indices = torch.topk(dist_values, k = k, largest = False).indices

        edge_index = []
        edge_weight = []

        locs_norm = F.normalize(locs, p=2, dim=1)  # (N, 2)
        sim_matrix = torch.matmul(locs_norm, locs_norm.T)

        for idx in top_indices:
            i, j = unique_node_pair[0][idx].item(), unique_node_pair[1][idx].item()
            edge_index.append([i, j])
            edge_index.append([j, i])  # undirected
            sim = sim_matrix[i, j].item()
            edge_weight.append(sim)
            edge_weight.append(sim)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        graphs.append(Data(x = node_features, edge_index = edge_index, edge_weight = edge_weight))

    return graphs


class Encoder_ST(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device, loc_dim, cycle_len, CosSin = False):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        self.attn = Attn_Fuse(d = d_model, nhead = n_head, hidden = 128, dropout = 0.5)

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb_temporal = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.event_emb_loc = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        # Temporal query 
        self.cycle_len = cycle_len
        self.tq_s  = nn.Parameter(torch.zeros(self.cycle_len, self.loc_dim), requires_grad=True)
        self.tq_t  = nn.Parameter(torch.zeros(self.cycle_len, 1), requires_grad=True)

        self.tq_s_emb = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.tq_t_emb = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        with torch.no_grad():
            for p in (self.tq_s, self.tq_t):
                p.normal_(mean=0.0, std=0.02)
        
        self.cnn = CNN(dim_out= 64)

    def build_tq(self, event_time, non_pad_mask):
        """
        event_time: [B, L]
        non_pad_mask: [B, L, 1]
        """
        phase_idx = torch.floor(event_time).to(torch.long) % self.cycle_len
        phase_idx = phase_idx.masked_fill(non_pad_mask.squeeze(-1) == 0, 0)

        tq_s  = self.tq_s[phase_idx]
        tq_t  = self.tq_t[phase_idx]
        
        return tq_s, tq_t

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask, pis = None):
        """ Encode event sequences via masked self-attention. """

        tq_s, tq_t = self.build_tq(event_time, non_pad_mask)
        tq_s = self.tq_s_emb(tq_s)
        tq_t = self.tq_t_emb(tq_t)
        tq_st = tq_s + tq_t

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)
        enc_output_loc = self.event_emb_loc(event_loc)
        if pis is None:
            enc_output = enc_output_temporal + enc_output_loc # [B, L, 64]
        else:
            e_topo = self.cnn(pis)
            enc_output = enc_output_temporal + enc_output_loc + 0.1 * self.attn(enc_output_loc, enc_output_temporal, e_topo)

        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                Q_theta=tq_s)

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                Q_theta=tq_t)

            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                Q_theta=tq_st)
        
        return enc_output, enc_output_temporal, enc_output_loc


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,device=None,loc_dim=2):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output = self.encoder(event_loc, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        return enc_output, non_pad_mask

class Transformer_ST(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024, n_layers=4, n_head=4, 
            d_k=64, d_v=64, dropout=0.1,device=None,loc_dim=2, cycle_len = 30, CosSin=False):
        super().__init__()

        self.encoder = Encoder_ST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            cycle_len = cycle_len,
            CosSin = CosSin
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time, pis = None):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output, enc_output_temporal, enc_output_loc = self.encoder(event_loc, event_time, non_pad_mask, pis)

        assert (enc_output != enc_output_temporal).any() & (enc_output != enc_output_loc).any() & (enc_output_loc != enc_output_temporal).any()
        
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)
        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output),dim=-1)

        return enc_output_all, non_pad_mask
    
class CNN(nn.Module):
    def __init__(self, dim_out):
        super(CNN, self).__init__()
        self.dim_out = dim_out
        self.features = nn.Sequential(
            nn.Conv2d(1, dim_out, kernel_size=3, stride=3), #channel of ZPI is 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),
        )
        self.maxpool = nn.MaxPool2d(5,5)

    def forward(self, zigzag_PI):
        feature = self.features(zigzag_PI)
        feature = self.maxpool(feature)
        feature = feature.view(-1, self.dim_out) #B, dim_out
        return feature


def _pairwise_haversine_deg(locs_deg: torch.Tensor) -> torch.Tensor:
    """
    locs_deg: [N, 2] as (lat_deg, lon_deg)
    returns: [N, N] pairwise distances in meters (diag = inf)
    """
    lat = torch.deg2rad(locs_deg[:, 0]).unsqueeze(1)  # [N,1]
    lon = torch.deg2rad(locs_deg[:, 1]).unsqueeze(1)
    dlat = lat - lat.t()                              # [N,N]
    dlon = lon - lon.t()
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat) @ torch.cos(lat).t() * torch.sin(dlon / 2) ** 2
    # numerical clamp guards
    c = 2 * torch.arcsin(torch.clamp(a.sqrt(), max=1.0 - 1e-12))
    R = 6_371_000.0  # Earth radius in meters
    dist = R * c
    dist.fill_diagonal_(float('inf'))
    return dist

def build_graph_haversine(event_time, event_loc_feat, event_loc_geo_deg, connect_ratio: float = 0.1):
    """
    event_time:        list[Tensor] length M, each [N_i]              (Δt_norm)
    event_loc_feat:    list[Tensor] length M, each [N_i, 2]           (lat_norm, lon_norm) for FEATURES
    event_loc_geo_deg: list[Tensor] length M, each [N_i, 2]           (lat_deg, lon_deg)   for EDGES
    connect_ratio:     fraction of unique pairs to keep (0,1]
    """
    graphs = []
    for times, loc_feat, loc_geo in zip(event_time, event_loc_feat, event_loc_geo_deg):
        N = loc_feat.size(0)
        assert times.ndim == 1 and times.size(0) == N
        assert loc_feat.shape == (N, 2)
        assert loc_geo.shape == (N, 2)

        # Node features: [Δt_norm, lat_norm, lon_norm]
        x = torch.cat([times.unsqueeze(1), loc_feat], dim=1)  # [N, 3]

        # Pairwise geodesic distance on raw coords, choose closest pairs
        D = _pairwise_haversine_deg(loc_geo)                  # [N, N], meters
        upper = torch.triu_indices(N, N, offset=1)            # unique unordered pairs
        dist_vals = D[upper[0], upper[1]]

        k = max(1, int(connect_ratio * dist_vals.numel()))    # keep at least 1
        topk = torch.topk(dist_vals, k=k, largest=False)
        i_u, j_u = upper[0][topk.indices], upper[1][topk.indices]

        # Undirected edge_index
        edge_index = torch.stack([torch.cat([i_u, j_u]), torch.cat([j_u, i_u])], dim=0)  # [2, 2k]

        graphs.append(Data(x=x, edge_index=edge_index))

    return graphs
