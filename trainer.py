import torch
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
from torch.nn.functional import log_softmax
from typing import List, Callable, Dict
import numpy as np
from typing import List, Callable
import gc
from collections import deque

class RLTrainer:
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 5e-6,
        max_length: int = 128,
        gradient_accumulation_steps: int = 4,
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
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.optimizer = bnb.optim.Adam8bit(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-7,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=1000,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Reward Normalization
        self.reward_ema = 0.0
        self.reward_std_ema = 1.0
        self.ema_alpha = 0.95

        self.loss_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)

    def safe_log_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True).clamp(min=1e-5)
        return logits - max_logits - torch.log(sum_exp)

    def generate_samples(
        self,
        prompts: List[str],
        num_samples: int = 1,
        temperature: float = 0.7
    ) -> List[str]:
        all_samples = []
        chunk_size = 4
        
        for i in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[i:i + chunk_size]
            inputs = self.tokenizer(chunk_prompts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=self.max_length,
                        num_return_sequences=num_samples,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        top_k=50,
                        top_p=0.95,
                        no_repeat_ngram_size=3
                    )
                
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_samples.extend(decoded)
            
            except RuntimeError as e:
                print(f"Warning: Generation failed for chunk {i}. Error: {str(e)}")
                all_samples.extend(["" for _ in range(len(chunk_prompts) * num_samples)])
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
            
        return all_samples

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        
        self.reward_ema = self.ema_alpha * self.reward_ema + (1 - self.ema_alpha) * batch_mean
        self.reward_std_ema = self.ema_alpha * self.reward_std_ema + (1 - self.ema_alpha) * batch_std
        
        normalized = (rewards - self.reward_ema) / (self.reward_std_ema + 1e-8)
        return torch.clamp(normalized, min=-4.0, max=4.0)

    def compute_loss(
        self,
        prompts: List[str],
        samples: List[str],
        rewards: torch.Tensor,
        chunk_size: int = 4
    ) -> Dict[str, torch.Tensor]:
        total_policy_loss = 0
        total_value_loss = 0
        valid_chunks = 0
        
        for i in range(0, len(samples), chunk_size):
            chunk_samples = samples[i:i + chunk_size]
            chunk_rewards = rewards[i:i + chunk_size]
            
            if not any(sample.strip() for sample in chunk_samples):
                continue
            
            inputs = self.tokenizer(chunk_samples, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            try:
                with torch.cuda.amp.autocast(enabled=True):
                    outputs = self.model(**inputs)
                    logits = outputs.logits.float()
                    
                    temperature = 1.0
                    scaled_logits = logits / temperature
                    log_probs = torch.log_softmax(scaled_logits[:, :-1, :], dim=-1)
                    
                    target_ids = inputs['input_ids'][:, 1:].to(self.device)
                    token_log_probs = log_probs.gather(
                        -1,
                        target_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    attention_mask = (target_ids != self.tokenizer.pad_token_id).float()
                    token_log_probs = token_log_probs * attention_mask
                    
                    advantages = chunk_rewards
                    
                    sequence_log_probs = token_log_probs.sum(dim=1)
                    policy_loss = -(sequence_log_probs * advantages).mean()
                    
                    entropy = -(torch.exp(log_probs) * log_probs).sum(dim=-1).mean()
                    entropy_bonus = 0.01 * entropy
                    
                    chunk_loss = policy_loss - entropy_bonus
                    
                    if not torch.isnan(chunk_loss) and not torch.isinf(chunk_loss):
                        total_policy_loss += chunk_loss
                        valid_chunks += 1
                        
                        self.loss_history.append(chunk_loss.item())
                        self.reward_history.append(chunk_rewards.mean().item())
            
            except RuntimeError as e:
                print(f"Warning: Loss computation failed for chunk {i}. Error: {str(e)}")
                continue
            
            del outputs, logits, log_probs, token_log_probs
            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()
        
        if valid_chunks == 0:
            raise RuntimeError("No valid chunks found for loss computation")
        
        return {
            'policy_loss': total_policy_loss / valid_chunks,
            'entropy': entropy,
            'mean_reward': sum(self.reward_history) / len(self.reward_history)
        }

    def train(
        self,
        train_prompts: List[str],
        rubric_fn: Callable[[str], float],
        num_epochs: int = 20,
        batch_size: int = 8,
        num_samples_per_prompt: int = 2,
        early_stopping_patience: int = 5
    ):
        print("Training with INT8 precision...")
        
        best_loss = float('inf')
        patience_counter = 0
        
        try:
            for epoch in range(num_epochs):
                total_loss = 0
                num_batches = 0
                epoch_rewards = []
                
                for i in range(0, len(train_prompts), batch_size):
                    batch_prompts = train_prompts[i:i + batch_size]
                    
                    try:
                        samples = self.generate_samples(batch_prompts, num_samples_per_prompt)
                        rewards = torch.tensor([
                            rubric_fn(s, p) 
                            for s, p in zip(samples, batch_prompts * num_samples_per_prompt)
                        ], device=self.device, dtype=torch.float32)
                        
                        normalized_rewards = self.normalize_rewards(rewards)
                        epoch_rewards.extend(rewards.cpu().tolist())
                        
                        metrics = self.compute_loss(batch_prompts, samples, normalized_rewards)
                        loss = metrics['policy_loss']
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        
                        if (num_batches + 1) % self.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                            self.scheduler.step()
                            self.optimizer.zero_grad()
                        
                        total_loss += loss.item()
                        num_batches += 1
                        
                    except Exception as e:
                        print(f"Warning: Batch processing failed. Error: {str(e)}")
                        continue
                
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
                reward_std = np.std(epoch_rewards) if epoch_rewards else 0
                
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Average Reward: {avg_reward:.4f} ± {reward_std:.4f}")
                print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.2e}")
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.save_checkpoint("best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print("Early stopping triggered!")
                        break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint...")
            self.save_checkpoint("interrupted_checkpoint.pt")
            
    def save_checkpoint(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)