import numpy as np
import torch
import torch.utils.data
import DSTPP.Constants as Constants


class EventData(torch.utils.data.Dataset):
    """ Event stream dataset. """
    def __init__(self, data):
        self.data = data
        self.length = len(data)

    def __len__(self):

        return self.length

    def __getitem__(self, idx):
        inst = self.data[idx]
        cols = list(zip(*inst))
        time, time_norm, lng, lat, *emb_list = [list(c) for c in cols]

        return (time, time_norm, lng, lat) + tuple(emb_list)


class EventDataWithPIS(EventData):
    def __init__(self, data, pis):
        super().__init__(data)
        self.pis = torch.as_tensor(pis, dtype=torch.float32)

    def __getitem__(self, idx):
        base = super().__getitem__(idx)

        return base + (self.pis[idx],)


def pad_time(insts):
    """ Pad the instance to the max seq length in batch. """

    max_len = max(len(inst) for inst in insts)
    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])
        
    return torch.tensor(batch_seq, dtype=torch.float32)


def collate_fn(insts):
    time, time_norm, lng, lat, *emb_cols = zip(*insts)

    time      = pad_time(time)
    time_norm = pad_time(time_norm)
    lng       = pad_time(lng)
    lat       = pad_time(lat)

    emb_list = [pad_time(e) for e in emb_cols]

    return (time, time_norm, lng, lat, *emb_list)

def collate_fn_with_pis(insts):
    time, time_norm, lng, lat, *emb_cols, pis = zip(*insts)

    time      = pad_time(time)
    time_norm = pad_time(time_norm)
    lng       = pad_time(lng)
    lat       = pad_time(lat)

    emb_list = [pad_time(e) for e in emb_cols]  # 8 tensors (B, T)
    pis      = torch.stack(pis, dim=0).to(torch.float32)

    return (time, time_norm, lng, lat, *emb_list, pis)

def get_dataloader(data, batch_size, shuffle = True, pis = None):
    
    if pis is None:
        ds = EventData(data)
        collate = collate_fn
    else:
        ds = EventDataWithPIS(data, pis)
        collate = collate_fn_with_pis

    return torch.utils.data.DataLoader(
        ds,
        num_workers=2,
        batch_size=batch_size,
        collate_fn=collate,
        shuffle=shuffle
    )
