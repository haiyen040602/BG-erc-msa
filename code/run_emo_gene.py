from emotional_gene import emotional_gene
from transformers import GPT2Tokenizer, GPT2LMHeadModel

model_name = "gpt2-medium"
model = GPT2LMHeadModel.from_pretrained(model_name, output_hidden_states=True)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

generated_text = emotional_gene(Knob=0.5, Prompt="also I", Topic='negative', Affect= 'anger', model=model, tokenizer=tokenizer)

print(generated_text)