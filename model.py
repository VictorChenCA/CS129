from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# Load and preprocess MATH-500
raw_dataset = load_dataset("HuggingFaceH4/MATH-500")['train']
def preprocess(example):
    problem = example['problem'].strip()
    answer = example['answer'].strip()
    return {'problem': problem, 'answer': answer}
dataset = raw_dataset.map(preprocess)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def extract_final_answer(output_text):
    # Implement a robust extraction based on your output format
    # For example, look for the last line or a specific marker
    return output_text.strip().split('\n')[-1]

def compute_reward(predicted_answer, gold_answer):
    # Simple string match; can be improved with normalization
    return int(predicted_answer == gold_answer)

def dapo_rl_step(model, tokenizer, batch, K=4, temperature=1.0):
    """
    batch: list of dicts with 'problem' and 'answer'
    """
    losses = []
    for example in batch:
        input_text = example['problem']
        gold_answer = example['answer']
        input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(model.device)
        # Sample K outputs
        outputs = []
        for _ in range(K):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=128,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id
            )
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output_text)
        # Compute rewards
        rewards = []
        for out in outputs:
            pred_answer = extract_final_answer(out)
            reward = compute_reward(pred_answer, gold_answer)
            rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=model.device)
        # Compute group-relative advantages
        mean_reward = rewards.mean()
        advantages = rewards - mean_reward
        # Compute log-probs for each output
        log_probs = []
        for out in outputs:
            # Tokenize full input+output
            full_text = input_text + "\n" + out
            tokens = tokenizer(full_text, return_tensors='pt').input_ids.to(model.device)
            with torch.no_grad():
                logits = model(tokens)  # (batch, seq, vocab)
            # Compute log-prob of output tokens
            # (You may need to align input/output tokens carefully)
            # For simplicity, assume output tokens start after input
            output_token_ids = tokens[0][input_ids.shape[-1]:]
            output_logits = logits.logits[0, input_ids.shape[-1]-1:-1, :]
            log_prob = F.log_softmax(output_logits, dim=-1)
            output_log_prob = log_prob.gather(1, output_token_ids.unsqueeze(-1)).sum()
            log_probs.append(output_log_prob)
        log_probs = torch.stack(log_probs)
        # DAPO loss: -sum(advantage * log_prob)
        loss = -(advantages * log_probs).mean()
        losses.append(loss)
    # Backprop
    total_loss = torch.stack(losses).mean()
    total_loss.backward()
    return total_loss.item()

batch_size = 4
num_epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = dapo_rl_step(model, tokenizer, batch, K=4)
        optimizer.step()
        print(f"Loss: {loss:.4f}")
