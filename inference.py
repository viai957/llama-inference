## Inference
from typing import Optional
import torch
import time
import json 
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from llama_model import ModelArgs, Transformer


class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, "No checkpoints files found"
            chk_path = checkpoints[0]
            print(f'Loaded checkpoint {chk_path}')
            checkpoint = torch.load(chk_path, map_location="cpu")
            print(f'Loaded checkpoint in {(time.time() - prev_time):.2f} seconds')
            prev_time = time.time()

        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len = max_seq_len,
            max_batch_size = max_batch_size,
            device = device,
            **params
        )
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            # Remove repe.freq from checkpoint as we are precomputiong the frequencies
            del checkpoint["rope.freqs"]
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded state dict in {(time.time() - prev_time):.2f} seconds")

        return LLaMA(model, tokenizer, model_args)
    
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        # Convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        # Make sure the batch size is not too large
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"Batch size {batch_size} is too large"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        # Make sure the prompt length is not larger than the maximum seq length
        assert max_prompt_len < self.args.max_seq_len, f"Prompt length {max_prompt_len} is too large"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list that will contain the generated tokens, along with the initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=self.args.device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=self.args.device)

        eos_reached = torch.tensor([False] * batch_size, device=self.args.device)
        prompt_tokens_mask = tokens != pad_id  # True if the token is a prompt token, False otherwise
        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature > 0:
                # The temperature is applied BEFORE the softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedily select the token with the maximum probability
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # EOS is reached only if we found an EOS token for padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # Cut to the EOS token, if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

            
    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim= -1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token



if __name__ == '__main__':
    import os
    torch.manual_seed(0)
    prompts = [
        "Simply put, the theory of relativity states that",
        "IF Google was an Indian company founded in Karnataka, it would"
        # Few shot prompt
        """Translate English to kananda:
        water : ನೀರು
        land : ಭೂಮಿ
        dusk : ಸಂಜೆ
        dawn : ಬೆಳಗುವಿಕೆ
        milk : ಹಾಲು"""
        # Zero shot prompt
        """Tell me if the following person is actually a real person or a fictional character:
        Name : Vignesh 
        Decision:
        """
    ]
    allow_cuda = True if 'CUDA_VISIBLE_DEVICES' in os.environ else False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    
    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='llama-2-7b/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    print('ALL OK')

    # Inferencing the model
    print("Inferenceing the model")
    out_tokens, out_text = model.text_completion(prompts, max_gen_len=64)
    assert len(out_tokens) == len(prompts)
    for i in range(len(out_text)):
        print(f"{out_text[i]}")
        print('-' * 50)