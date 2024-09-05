import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

class MonolingualDataset(Dataset):

    def __init__(self, ds, tokenizer, seq_len, truncation=True) -> None:
        super().__init__()

        self.ds = ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer.bos_token_id], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.eos_token_id], dtype=torch.int64)
        self.truncation = truncation

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        text = self.ds[index]

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        num_padding = self.seq_len - len(tokens) - 1

        if num_padding < 0:
            if self.truncation is True:
                tokens = tokens[:self.seq_len-1]
                num_padding = 0
            else:
                raise ValueError('Sentence is too long')

        input_tokens = torch.cat([
                                self.sos_token,
                                torch.tensor(tokens, dtype=torch.int64),
                                torch.tensor([self.eos_token] * num_padding, dtype=torch.int64)
                                ])

        label = torch.cat([
                        torch.tensor(tokens, dtype=torch.int64),
                        self.eos_token,
                        torch.tensor([self.eos_token] * num_padding, dtype=torch.int64)
                        ])

        assert input_tokens.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
                "input_tokens": input_tokens, # (seq_len)
                "mask" : (input_tokens != self.eos_token).int(), # (1, Seq_len) 
                "label" : label, # (seq_len)
                'text': text,
                 }

def get_ds(config, tokenizer, split='train'):
    if config['dataset'] == "stanfordnlp/imdb":
        ds = load_dataset(config['dataset'])[split]['text']
    else:
        ds = load_dataset(config['dataset'], config['subset'])[split]['text']

    if split == 'train' and config['train_size'] != 1:
        idx = int(len(ds) * config['train_size'])
        ds = ds[:idx]

    ds = [x for x in ds if x.strip()] # many empty strings in this dataset

    ds = MonolingualDataset(ds, tokenizer, config['seq_len'])
    if split == 'train':
        dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
    else:
        dataloader = DataLoader(ds, batch_size=config['batch_size'], shuffle=False)

    return dataloader
