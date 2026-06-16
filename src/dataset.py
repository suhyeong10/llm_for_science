import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PretrainDataset(Dataset):
    def __init__(self, file_path="data/cpt_sample.txt", block_size=512):
        self.block_size = block_size
        
        # 1. 토크나이저 준비
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token 

        # 2. ⭐ [진짜 데이터를 불러오는 핵심 지점] ⭐
        # 하드디스크에 저장한 cpt_sample.txt 파일을 열어서 텍스트를 메모리로 로드합니다.
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 3. 글자를 숫자로 바꾸고 블록 크기(512)로 쪼개기
        tokenized_text = self.tokenizer.encode(text)
        self.examples = []
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):
            self.examples.append(tokenized_text[i : i + block_size])
        
        if not self.examples:
            padded = tokenized_text + [self.tokenizer.pad_token_id] * (block_size - len(tokenized_text))
            self.examples.append(padded[:block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        token_ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {
            "input_ids": token_ids,
            "labels": token_ids
        }
