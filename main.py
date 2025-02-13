# main.py
import shutil

import pandas as pd
import polars as pl
import argparse
from predictor import Predictor

def main():
    parser = argparse.ArgumentParser(description="ChainPatch: 自动生成 Git 补丁")
    parser.add_argument("--problem", type=str, required=True, help="问题描述（问题语句）")
    parser.add_argument("--archive", type=str, required=True, help="代码仓库压缩包路径（.tar 文件）")
    parser.add_argument("--model", type=str, default="deepseek-r1", help="模型路径或名称")
    args = parser.parse_args()

    predictor = Predictor(model_path=args.model)
    patch = predictor.predict(args.problem, args.archive)
    if patch:
        print("Generated Patch:\n", patch)
    else:
        print("No valid patch generated.")

if __name__ == "__main__":
    main()
