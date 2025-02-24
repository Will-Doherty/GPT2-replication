from datasets import load_dataset
from model_def import ModelConfig

def transform_data(example):
    tokens = ModelConfig.tokenizer.encode(example["text"]) + [50256]
    return {"input_ids": tokens[:-1], "labels": tokens[1:]}

def get_data():
    fine_web = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )
    dataset = fine_web.map(transform_data, remove_columns=fine_web.column_names)
    torch_dataset = dataset.with_format("torch")
    return torch_dataset