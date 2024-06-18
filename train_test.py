import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 데이터셋 클래스 정의
class MathDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        problem = self.dataframe.iloc[idx, 1]  # 문제
        answer = str(self.dataframe.iloc[idx, 2])  # 정답을 문자열로 변환
        
        encoding = self.tokenizer(
            problem,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            answer,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        labels = target_encoding['input_ids'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

# 데이터셋 로드 및 전처리
file_path = './train.csv'
train_data = pd.read_csv(file_path)

# 모델과 토크나이저 초기화
model_name = "./deepseek-math"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 패딩 토큰 추가
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 데이터셋 생성
train_dataset = MathDataset(train_data, tokenizer)

# 훈련 설정 정의
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 배치 크기를 줄임
    per_device_eval_batch_size=1,   # 배치 크기를 줄임
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    fp16=True,  # Mixed Precision Training 활성화
    gradient_accumulation_steps=4,  # 그래디언트 누적 단계 수
)

# Trainer 초기화
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# 모델 학습
trainer.train()

# 모델 저장
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
