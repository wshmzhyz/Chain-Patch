# predictor.py
import os
import shutil
from utils import Utils
from file_query import FileQuery
from patch_generator import PatchGenerator
from patch_verifier import PatchVerifier
from llm_provider import LLMProvider

REPO_PATH = "repo"

class Predictor:
    def __init__(self, model_path="deepseek-r1"):
        self.llm_provider = LLMProvider()
        self.file_query = FileQuery(self.llm_provider)
        self.patch_generator = PatchGenerator(self.llm_provider)
        self.patch_verifier = PatchVerifier(self.llm_provider)

    def predict_inner(self, problem_statement: str, directory: str) -> str:
        # 获取目录字符串
        directory_string = Utils.stringify_directory(directory)
        # 获取文件查询结果（即哪些文件需要检查以及搜索关键字）
        file_query, query_response = self.file_query.get_query(directory_string, problem_statement)
        print("File Query Response:\n", query_response)
        print("Extracted File Query:", file_query)
        if not file_query:
            return None
        # 根据文件查询结果提取文件内容
        file_content_string = Utils.fetch_file_contents(file_query, repo_path=REPO_PATH)
        print("Fetched File Contents:\n", file_content_string)
        # 多阶段候选生成与验证
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"Generating candidate patch, attempt {attempt + 1}")
            candidate_patch, patch_response = self.patch_generator.get_patch(problem_statement, file_content_string)
            print("Candidate Patch Response:\n", patch_response)
            print("Candidate Patch:\n", candidate_patch)
            if not candidate_patch:
                continue
            verified_patch, verify_response = self.patch_verifier.verify_patch(problem_statement, file_content_string, candidate_patch)
            if verified_patch is not None:
                print(f"Candidate patch accepted on attempt {attempt + 1}")
                return verified_patch
            else:
                print(f"Candidate patch rejected on attempt {attempt + 1}")
                print("Verification Response:\n", verify_response)
        return None

    def predict(self, problem_statement: str, repo_archive_path: str) -> str:
        """
        1. 将 repo_archive_path 解压到 REPO_PATH。
        2. 调用 predict_inner 获取补丁。
        3. 清理解压目录，返回生成的补丁字符串（或 None）。
        """
        if os.path.exists(REPO_PATH):
            shutil.rmtree(REPO_PATH)
        shutil.unpack_archive(repo_archive_path, extract_dir=REPO_PATH)
        patch_string = self.predict_inner(problem_statement, REPO_PATH)
        shutil.rmtree(REPO_PATH)
        return patch_string
