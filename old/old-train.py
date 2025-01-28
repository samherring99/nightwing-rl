import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn.functional import log_softmax
import numpy as np
from typing import List, Callable

### Needs 8bit Adam setup

class RLTrainer:
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        max_length: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.max_length = max_length

    def generate_samples(
        self,
        prompts: List[str],
        num_samples: int = 1,
        temperature: float = 1.0
    ) -> List[str]:
        all_samples = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_return_sequences=num_samples,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_samples.extend(decoded)
            
        return all_samples

    def compute_rewards(
        self,
        samples: List[str],
        rubric_fn: Callable[[str], float],
        prompts: List[str]
    ) -> torch.Tensor:
        rewards = []
        
        for sample, prompt in zip(samples, prompts * len(samples)):
            generated_text = sample[len(prompt):]
            reward = rubric_fn(generated_text, prompt)
            rewards.append(reward)
            
        return torch.tensor(rewards, device=self.device)

    def compute_loss(
        self,
        prompts: List[str],
        samples: List[str],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        loss = 0
        batch_size = len(prompts)
        
        for i, (prompt, sample, reward) in enumerate(zip(prompts * len(samples), samples, rewards)):
            inputs = self.tokenizer(sample, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            log_probs = log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = inputs.input_ids[:, 1:]
            token_log_probs = log_probs.gather(
                -1,
                target_ids.unsqueeze(-1)
            ).squeeze(-1)
            sequence_log_prob = token_log_probs.sum()
            loss -= sequence_log_prob * reward

        return loss / batch_size

    def train_step(
        self,
        prompts: List[str],
        rubric_fn: Callable[[str], float],
        num_samples: int = 1
    ) -> float:

        samples = self.generate_samples(prompts, num_samples)
        rewards = self.compute_rewards(samples, rubric_fn, prompts)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        loss = self.compute_loss(prompts, samples, rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(
        self,
        train_prompts: List[str],
        rubric_fn: Callable[[str], float],
        num_epochs: int = 10,
        batch_size: int = 8,
        num_samples_per_prompt: int = 1
    ):
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(train_prompts), batch_size):
                batch_prompts = train_prompts[i:i + batch_size]
                
                loss = self.train_step(
                    batch_prompts,
                    rubric_fn,
                    num_samples_per_prompt
                )
                
                total_loss += loss
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")