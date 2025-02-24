import tiktoken
import torch
from tqdm import tqdm
import torch.nn.functional as F
from datasets import load_dataset
from model_def import ModelConfig, Transformer
from train import TrainingConfig

tokenizer = tiktoken.get_encoding("gpt2")
device = ModelConfig.device

def prepare_example(example):
    ctx, endings = example["ctx"], example["endings"]
    ctx_ids = tokenizer.encode(ctx)  # .encode returns a list of token ids when given a string

    all_combined_ids = []
    all_masks = []
    for ending in endings:
        ending_ids = tokenizer.encode(" " + ending)
        combined_ids = ctx_ids + ending_ids
        mask = [0]*len(ctx_ids) + [1]*len(ending_ids)
        all_combined_ids.append(combined_ids)
        all_masks.append(mask)

    max_len = max((len(ids) for ids in all_combined_ids))
    tokens = torch.zeros((4, max_len), dtype=torch.int64)
    mask = torch.zeros((4, max_len), dtype=torch.int64)
    for i, (token_row, mask_row) in enumerate(zip(all_combined_ids, all_masks)):
        tokens[i, :len(token_row)] = torch.tensor(token_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return mask, tokens

@torch.no_grad()
def score_completion(model, example, hf_model: bool):
    mask, tokens = prepare_example(example)
    targets = tokens[:, 1:].contiguous()
    tokens = tokens[:, :-1].contiguous()
    mask = mask[:, :-1].contiguous()
    tokens, targets, mask = tokens.to(device), targets.to(device), mask.to(device)
    output = model(tokens)
    if hf_model:
        output = output["logits"]
    scores = F.cross_entropy(output[..., :tokenizer.n_vocab].view(-1, tokenizer.n_vocab), targets.view(-1), reduction='none')
    masked_scores = mask * scores.view(4, -1)
    average_scores = torch.mean(masked_scores, dim=1, keepdim=True)
    prediction = average_scores.argmin()
    return prediction

def test_model_on_hellaswag(model, dataset, device="cuda", hf_model=False):
    model.eval()
    model.to(device)
    # model = torch.compile(model)
    num_correct = 0
    for example in tqdm(dataset):
        prediction = score_completion(model, example, hf_model)
        label = torch.tensor(int(example["label"]), dtype=torch.int64)
        if prediction == label:
            num_correct += 1
    return num_correct / len(dataset)

if __name__ == '__main__':
    hellaswag = load_dataset("Rowan/hellaswag")
    dset = hellaswag["validation"]
    model = Transformer()
    model.load_state_dict(torch.load(TrainingConfig.model_save_path))
    prop_correct = test_model_on_hellaswag(model=model, dataset=dset)
    print(f"\nProportion of questions answered correctly = {prop_correct:.1%}")