from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import matplotlib.pyplot as plt

import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset_filename = "alpaca_data_translated.json"
dataset = load_dataset('json', data_files=dataset_filename)

train_data = dataset['train'].train_test_split(test_size=0.2)
train_dataset, val_dataset = train_data['train'], train_data['test']

def tokenize_function(example):
    prompt = example['instruction']
    inputs = tokenizer(prompt, max_length=512, truncation=True, padding="max_length")
    outputs = tokenizer(example['output'], max_length=512, truncation=True, padding="max_length")
    inputs["labels"] = outputs["input_ids"]    
    return inputs


tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    no_cuda=False
)



trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer
)

trainer.train()

train_loss = trainer.state.log_history

plt.plot([x['loss'] for x in train_loss if 'loss' in x])
plt.title('Training Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.show()
