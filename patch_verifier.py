# patch_verifier.py
from llm_provider import LLMProvider

class PatchVerifier:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.tokenizer = llm_provider.tokenizer
        self.verifying_prompt = """
Before evaluating the following patch, consider this example:

Example:
Problem Statement: "Pylint reports a TypeError due to formatting a None value."
Relevant file snippet:
    formatted = format(value.value, format_spec.value)
Proposed Patch Example:
<patch>
--- a/astroid/nodes/node_classes.py
+++ b/astroid/nodes/node_classes.py
@@ -X,Y +X,Y @@
-    formatted = format(value.value, format_spec.value)
+    formatted = format(value.value if value.value is not None else 0, format_spec.value)
</patch>
Chain of Thought:
1. The patch checks if value.value is None before formatting.
2. This should prevent the TypeError.
Observation: The patch appears to address the issue by substituting a default value.
Expected final evaluation: <label>Yes</label>, this fixes the problem.

Now, evaluate the following patch:

This is the problem statement.

{problem_statement}

These are the files that are thought to be relevant, which may not be complete.

{file_content_string}

This is the proposed patch to fix the problem.

{patch_string}

Firstly, list your observations.
Then, evaluate whether the patch fully fixes the problem described in the problem statement.

End your response with exactly either of:
- <label>Yes</label>, this fixes the problem.
- <label>No</label>, this does not fix the problem.

Note:
- Only evaluate; do not provide suggestions on how to fix.
- End with exactly either <label>Yes</label> or <label>No</label>.
""".strip()

    def verify_patch(self, problem_statement, file_content_string, patch_string):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=32768,
        )
        prompt = self.verifying_prompt.format(
            problem_statement=problem_statement,
            file_content_string=file_content_string,
            patch_string=patch_string
        )
        # 生成多个回复，进行投票判断
        list_of_messages = [[{"role": "user", "content": prompt}] for _ in range(4)]
        list_of_texts = [
            self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
            for messages in list_of_messages
        ]
        print("PatchVerifier token lengths:", [len(self.tokenizer.encode(text)) for text in list_of_texts])
        responses = self.llm_provider.generate(prompts=list_of_texts, sampling_params=sampling_params)
        if not responses:
            return None, ""
        response_texts = [resp.outputs[0].text for resp in responses]
        # 如果所有回复均包含 <label>Yes</label>，则认为补丁有效
        for text in response_texts:
            if "<label>Yes</label>" not in text:
                return None, text
        return patch_string, response_texts[-1]
