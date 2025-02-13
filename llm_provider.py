# llm_provider.py
import os
from vllm import LLM, SamplingParams
import warnings

warnings.simplefilter('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_pth = '../input/m/deepseek-r1/transformers/deepseek-r1-distill-qwen-32b-awq/1'

MAX_NUM_SEQS = 4
MAX_MODEL_LEN = 32768
MAX_TOKENS  = 32768

class LLMProvider:
    def __init__(self,   tensor_parallel_size=1, gpu_memory_utilization=0.9, seed=42):
        self.llm = LLM(
            model_path=llm_model_pth,
            max_num_seqs=MAX_NUM_SEQS,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(self, prompts, sampling_params):
        return self.llm.generate(prompts=prompts, sampling_params=sampling_params)
