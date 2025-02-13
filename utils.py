# utils.py
import os
from io import StringIO

class Utils:
    @staticmethod
    def stringify_directory(directory):
        """遍历目录，返回所有文件的完整路径（每行一个）。"""
        full_paths = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                full_paths.append(full_path)
        return "\n".join(full_paths)

    @staticmethod
    def fetch_file_contents(files_to_search, repo_path="repo", context_lines=10, max_gap=0):
        """
        根据文件查询结果（字典：filepath -> [搜索字符串]），读取各文件并提取匹配内容，
        返回格式化后的内容字符串。
        """
        def find_lines_in_files_with_context(search_map, context_lines=context_lines):
            all_matches_per_file = []
            for path, terms in search_map.items():
                if not os.path.isfile(path):
                    all_matches_per_file.append([])
                    continue
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                file_snippets = []
                num_lines = len(lines)
                for i, line in enumerate(lines, start=1):
                    if any(t in line for t in terms):
                        start_idx = max(1, i - context_lines)
                        end_idx = min(num_lines, i + context_lines)
                        snippet = []
                        for snippet_no in range(start_idx, end_idx + 1):
                            text_content = lines[snippet_no - 1].rstrip("\n")
                            snippet.append((snippet_no, text_content))
                        file_snippets.append(snippet)
                all_matches_per_file.append(file_snippets)
            return all_matches_per_file

        def merge_file_snippets(file_snippets, gap=0):
            intervals = []
            for snippet in file_snippets:
                if snippet:
                    start_line = snippet[0][0]
                    end_line = snippet[-1][0]
                    intervals.append((start_line, end_line, snippet))
            intervals.sort(key=lambda x: x[0])
            merged = []
            for start, end, snippet in intervals:
                if not merged:
                    merged.append((start, end, snippet))
                    continue
                prev_start, prev_end, prev_snippet = merged[-1]
                if start <= prev_end + gap:
                    new_end = max(end, prev_end)
                    combined_dict = {}
                    for ln, txt in prev_snippet:
                        combined_dict[ln] = txt
                    for ln, txt in snippet:
                        combined_dict[ln] = txt
                    merged_snippet = [(ln, combined_dict[ln]) for ln in sorted(combined_dict)]
                    merged[-1] = (prev_start, new_end, merged_snippet)
                else:
                    merged.append((start, end, snippet))
            return [x[2] for x in merged]

        def merge_all_snippets(all_files_snips, gap=0):
            merged = []
            for snips in all_files_snips:
                merged.append(merge_file_snippets(snips, gap=gap))
            return merged

        context_snippets = find_lines_in_files_with_context(files_to_search, context_lines=context_lines)
        merged_snips = merge_all_snippets(context_snippets, gap=max_gap)

        output = StringIO()
        output.write("Sample files created successfully.\n\n")
        output.write("Search Results (by file, merging any overlapping context):\n\n")
        for (filepath, snippet_list) in zip(files_to_search.keys(), merged_snips):
            output.write(f"FILE: {filepath[len(repo_path) + 1:]}\n")
            output.write("-" * 60 + "\n")
            if not snippet_list:
                output.write("  No matches found.\n")
            else:
                for snippet_idx, snippet in enumerate(snippet_list, start=1):
                    snippet_start = snippet[0][0]
                    snippet_end = snippet[-1][0]
                    output.write(f"Match #{snippet_idx}, lines {snippet_start} to {snippet_end}:\n")
                    for line_no, text in snippet:
                        output.write(f"  {line_no:3d} | {text}\n")
                    output.write("\n")
            output.write("=" * 60 + "\n\n")
        return output.getvalue()
