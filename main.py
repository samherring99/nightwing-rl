from trainer import RLTrainer

def example_rubric(generated_text: str, prompt: str) -> float:
    score = 0.0
    if len(generated_text.split()) >= 10:
        score += 0.2
        
    if generated_text.endswith(('.', '!', '?')):
        score += 0.8
    
    return score

prompts = ["Write a story about:", "Explain the concept of:"]

trainer = RLTrainer(
    "meta-llama/Llama-3.2-1B",
    learning_rate=1e-5,
)

trainer.train(
    train_prompts=prompts,
    rubric_fn=example_rubric,
    batch_size=32,
    num_samples_per_prompt=4
)