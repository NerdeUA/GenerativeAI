from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GenerationConfig
import torch

checkpoint_path = "./results/checkpoint-15603"
generation_config = GenerationConfig.from_pretrained(checkpoint_path)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_text(prompt, model, tokenizer, max_length=50, temperature=1.0, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, add_special_tokens=True).to(device)
    outputs = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        generation_config=generation_config,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Модель готова до роботи. Введіть запит або напишіть 'exit' для завершення.")
while True:
    prompt = input("Ваше питання: ")
    if prompt.lower() == "exit":
        print("Завершення роботи.")
        break
    try:
        response = generate_text(prompt, model, tokenizer)
        print("Відповідь:", response)
    except Exception as e:
        print("Сталася помилка:", str(e))
