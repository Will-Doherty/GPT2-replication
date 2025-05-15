from datasets import load_dataset
from model_def import ModelConfig
from torch.utils.data import IterableDataset
from collections import deque
import torch

class TokenChunkDataset(IterableDataset):
    def __init__(self, dataset_stream, tokenizer, seq_len, pad_token=50256):
        self.dataset_stream = dataset_stream
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token = pad_token

    def __iter__(self):
        buffer = deque()
        for example in self.dataset_stream:
            tokens = self.tokenizer.encode(example["text"]) + [self.pad_token]
            buffer.extend(tokens)
            while len(buffer) > self.seq_len:
                chunk = [buffer.popleft() for _ in range(self.seq_len + 1)]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels    = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}

def get_data(rank, world_size):
    fine_web = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    fine_web = fine_web.shard(
        num_shards=world_size,
        index=rank
    )
    dataset = TokenChunkDataset(fine_web, ModelConfig.tokenizer, ModelConfig.max_seq_len)
    return dataset