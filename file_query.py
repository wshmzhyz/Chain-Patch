# file_query.py
import re
import xml.etree.ElementTree as ET
from llm_provider import LLMProvider

class FileQuery:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider
        self.tokenizer = llm_provider.tokenizer
        self.reading_prompt = """
You will be implementing a git diff patch to solve an issue with the code repository.
Before selecting the files to inspect, think carefully about the problem and determine which parts of the code are most likely causing the issue.
Below is an example demonstrating the chain-of-thought process:

Example:
Problem Statement: "When running pylint on a.py, a TypeError occurs due to an unsupported format string on a NoneType value. The error originates in astroid's _infer_from_values."
File Directory:
<directory>
repo/astroid/nodes/node_classes.py
repo/astroid/transform.py
repo/astroid/context.py
repo/astroid/nodes/_base_nodes.py
repo/astroid/nodes/scoped_nodes/scoped_nodes.py
</directory>
Chain of Thought:
1. The error message indicates that a None value is being formatted, which is not allowed.
2. The file "repo/astroid/nodes/node_classes.py" contains the function _infer_from_values where "format(value.value, format_spec.value)" is called.
3. To pinpoint the problem, search for the strings "format_spec.value" and "value.value" in that file.
Final Selection:
<root>
    <entry>
        <filepath>repo/astroid/nodes/node_classes.py</filepath>
        <strings_to_search>
            <string_to_search>format_spec.value</string_to_search>
            <string_to_search>value.value</string_to_search>
        </strings_to_search>
    </entry>
</root>

Now, process the following:

This is the problem statement.

{problem_statement}

This is the file directory

<directory>
{directory_string}
</directory>

Which files should be inspected so that we can solve the problem?
When inspecting each file, what strings should be searched?

Return the strings to search in this format:

(explanation)

<root>
    <entry>
        <filepath>filepath</filepath>
        <strings_to_search>
            <string_to_search>string_to_search</string_to_search>
            ...
        </strings_to_search>
    </entry>
</root>

Notes:
- Ensure each entry is enclosed between <root> and </root>.
- Return the FULL filepath exactly as given in the directory.
- If searching for a function or keyword, consider adding surrounding spaces or punctuation (e.g., " calculate(").
- Prefer longer, more specific search strings.
- Do not inspect more than 5 files; include only the necessary ones.
""".strip()

    def get_query(self, directory_string, problem_statement):
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            temperature=1.0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=32768,
        )
        prompt = self.reading_prompt.format(
            problem_statement=problem_statement,
            directory_string=directory_string
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
        print("FileQuery token lengths:", [len(self.tokenizer.encode(text)) for text in list_of_texts])
        response = self.llm_provider.generate(prompts=list_of_texts, sampling_params=sampling_params)
        if not response:
            return "", ""
        response_text = response[0].outputs[0].text
        file_query = self.extract_file_query(response_text)
        return file_query, response_text

    @staticmethod
    def extract_file_query(xml_content):
        parsed_data = {}
        pattern = r'<root>(.*?)</root>'
        matches = re.findall(pattern, xml_content, re.DOTALL)
        for match in matches:
            try:
                root = ET.fromstring("<root>" + match + "</root>")
                for entry in root.findall("entry"):
                    filepath = entry.find("filepath")
                    filepath_text = filepath.text.strip() if filepath is not None else None
                    strings_container = entry.find("strings_to_search")
                    search_strings = []
                    if strings_container is not None:
                        for s in strings_container.findall("string_to_search"):
                            if s.text is not None:
                                search_strings.append(s.text.strip())
                    parsed_data[filepath_text] = search_strings
            except Exception as e:
                print("Error parsing XML:", e)
                return ""
        return parsed_data
