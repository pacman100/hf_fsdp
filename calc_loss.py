import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from together.modeling_flash_llama import LlamaForCausalLM
from copy import deepcopy
import gc

device = "cuda"
model_id = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


dataset = load_dataset("CarperAI/pilev2-dev", data_dir="data/pubmed", split='train[:1000]', num_proc=16)
tokenizer.pad_token = tokenizer.eos_token


def collate_fn(examples, debug=False):
    batch_size = len(examples)
    input_ids = tokenizer([e['text'] for e in examples], return_tensors='pt', padding='max_length', truncation=True, max_length=4096)['input_ids']
    labels = deepcopy(input_ids)
    labels[labels == tokenizer.pad_token_id] = -100
    # For HF style:
    batch = {'input_ids': input_ids.to(device), 'labels': labels.to(device)}
    return batch


batch_size = 4



train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    collate_fn=collate_fn,
    shuffle=True,
)

#@torch.compile()
def inference_func(batch, losses):
    with torch.no_grad():
        losses.append(model(**batch).loss.cpu())
        print(losses[-1], np.mean(losses))
        torch.cuda.empty_cache(); gc.collect()
    return losses


losses = []
for batch in tqdm(train_loader):
     losses = inference_func(batch,losses)

print('Loss: ', np.mean(losses))


