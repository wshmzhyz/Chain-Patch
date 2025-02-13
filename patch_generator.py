# patch_generator.py
import re
from llm_provider import LLMProvider

class PatchGenerator:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.tokenizer = llm_provider.tokenizer
        self.patching_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
Before writing the patch, carefully reason through the problem with the following example:

Example:
Problem Statement: "Pylint reports a TypeError in a.py because a NoneType is being formatted in astroid's _infer_from_values."
Relevant file snippet from repo/astroid/nodes/node_classes.py might show:
    formatted = format(value.value, format_spec.value)
Chain of Thought:
1. The error occurs because value.value is None.
2. A possible fix is to check if value.value is None and substitute a default (e.g., 0) or skip formatting.
3. Therefore, modify the code to use: formatted = format(value.value if value.value is not None else 0, format_spec.value)
Final Patch Example:
<patch>
--- a/astroid/nodes/node_classes.py
+++ b/astroid/nodes/node_classes.py
@@ -X,Y +X,Y @@
-    formatted = format(value.value, format_spec.value)
+    formatted = format(value.value if value.value is not None else 0, format_spec.value)
</patch>

Now, generate a patch for the following:

This is the problem statement.

{problem_statement}

These are the files that are thought to be relevant

{file_content_string}

Write a git diff within <patch> and </patch> that fixes the problem.
""".strip()

    def get_patch(self, problem_statement, file_content_string):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=32768,
        )
        prompt = self.patching_prompt.format(
            problem_statement=problem_statement,
            file_content_string=file_content_string
        )
        list_of_messages = [[{"role": "user", "content": prompt}]]
        list_of_texts = [
            self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in list_of_messages
        ]
        print("PatchGenerator token lengths:", [len(self.tokenizer.encode(text)) for text in list_of_texts])
        response = self.llm_provider.generate(prompts=list_of_texts, sampling_params=sampling_params)
        if not response:
            return "", ""
        response_text = response[0].outputs[0].text
        patch_string = self.extract_patch_string(response_text)
        return patch_string, response_text

    @staticmethod
    def extract_patch_string(text):
        pattern = r'<patch>(.*?)</patch>'
        matches = re.findall(pattern, text, re.DOTALL)
        if not matches:
            return None
        return "\n".join(matches)
