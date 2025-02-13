# fine_tuner.py
import os
import torch
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model


class FineTuner:
    def __init__(
            self,
            model_name="deepseek-r1",
            dataset_path="./input/train-00000-of-00001.parquet",
            output_dir="./fine_tuned_model",
            per_device_train_batch_size=4,
    ):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.per_device_train_batch_size = per_device_train_batch_size

        # QLoRA 4-bit 量化配置
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # LoRA 配置
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # 加载分词器和预训练模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=self.bnb_config, device_map="auto"
        )
        self.model = get_peft_model(self.model, self.lora_config)

    def load_data(self):
        # 使用 datasets 库加载 parquet 格式的数据集
        dataset = load_dataset("parquet", data_files=self.dataset_path)["train"]

        # 构造用于微调的输入文本。
        def preprocess(example):
            input_text = "Problem: " + example["problem_statement"] + " Patch: " + example["patch"]
            example["input_text"] = input_text
            return example

        dataset = dataset.map(preprocess)
        return dataset

    def data_collator(self, batch):
        # 将输入文本 token 化，并设置 labels
        inputs = [example["input_text"] for example in batch]
        tokenized = self.tokenizer(
            inputs, padding=True, truncation=True, max_length=1024, return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized

    def fine_tune(
            self,
            num_train_epochs=3,
            learning_rate=1e-5,
            max_steps=1000,
            logging_steps=10,
            save_steps=200,
    ):
        dataset = self.load_data()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            learning_rate=learning_rate,
            max_steps=max_steps,
            logging_steps=logging_steps,
            save_steps=save_steps,
            evaluation_strategy="no",
            fp16=True,
            report_to="none",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )
        trainer.train()
        # 保存微调后的模型和分词器
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print("Fine-tuning completed. Model saved to", self.output_dir)


if __name__ == "__main__":
    tuner = FineTuner()
    tuner.fine_tune()
