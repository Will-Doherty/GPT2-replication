from transformers import AutoModelForCausalLM
from datasets import load_dataset
from evaluate import test_model_on_hellaswag

if __name__ == '__main__':
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2")
    hellaswag = load_dataset("Rowan/hellaswag")
    dset = hellaswag["validation"]
    prop_correct = test_model_on_hellaswag(model=gpt2, dataset=dset, hf_model=True)
    print(f"\nProportion of questions answered correctly = {prop_correct:.1%}")