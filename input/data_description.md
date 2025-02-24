configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
    dataset_info:
    features:
  - name: instance_id
    dtype: string
  - name: patch
    dtype: string
  - name: repo
    dtype: string
  - name: base_commit
    dtype: string
  - name: hints_text
    dtype: string
  - name: test_patch
    dtype: string
  - name: problem_statement
    dtype: string
  - name: version
    dtype: int64
  - name: environment_setup_commit
    dtype: string
  - name: FAIL_TO_PASS
    sequence: string
  - name: PASS_TO_PASS
    sequence: string
  - name: meta
    struct:
    - name: failed_lite_validators
      sequence: string
    - name: has_test_patch
      dtype: bool
    - name: is_lite
      dtype: bool
  - name: created_at
    dtype: timestamp[ns, tz=UTC]
  - name: license
    dtype: string
    splits:
  - name: train
    num_bytes: 88219540
    num_examples: 6411
    download_size: 24592081
    dataset_size: 88219540
    license: cc-by-4.0
    tags:
- code
- synthetic
- tools
- agents
- software
size_categories:
- 1K<n<10K
---

# Dataset Summary
SWE-bench Extra is a dataset that can be used to train or evaluate agentic systems specializing in resolving GitHub issues. It is based on the methodology used to build SWE-bench benchmark and includes 6,415 Issue-Pull Request pairs sourced from 1,988 Python repositories.

# Dataset Description
The SWE-bench Extra dataset supports the development of software engineering agents capable of autonomously solving GitHub issues. The data collection process, based on the SWE-bench methodology, involves the following steps:

1. **Issue and Pull Request Collection**: Issues are gathered and linked with pull requests that successfully resolve them.
2. **Filtering**: Instances are filtered based on attributes such as issue descriptions, relevant code paths, and test patches.
3. **Execution-based Validation**: The project environments are set up and tests are run to verify that they execute correctly.

For a more detailed description of the data collection process, please refer to our blog post [Scaling data collection for training software engineering agents](https://nebius.com/blog/posts/scaling-data-collection-for-training-swe-agents).

As an example use case of this dataset, we’ve used SWE-bench-extra instances to generate a dataset of 80,036 trajectories [`nebius/swe-agent-trajectories`](https://huggingface.co/datasets/nebius/swe-agent-trajectories). We’ve then trained an action generator model, that achieves a score of 19.2% on the subset of 50 random instances from the SWE-bench Verified benchmark, representing a 30% relative improvement over its parent model Qwen2.5-72B-Instruct, which scored 14.8%. Further augmenting the action generator with a guided search based on a critic model, also trained on this data, achieves 40.6% on the full SWE-bench Verified benchmark, which is state-of-the-art among agents using solely open-weight models. You can read more about this agent in our blog post, [“Leveraging Training and Search for Better Software Engineering Agents”](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents).

# How to Use

```python
from datasets import load_dataset
ds = load_dataset('nebius/SWE-bench-extra')
```

# Dataset Statistics
Average, 75th percentile, and maximum values characterizing various attributes of the collected instances. Statistics are micro-averaged without grouping by repository.

| Data          | Type               | Mean     | p75      | Max       |
|---------------|--------------------|----------|----------|-----------|
| Issue text    | Length (words)     | 111.5    | 146      | 1,294     |
| Code base     | Files (Non-test)   | 71.71    | 72.00    | 2,264     |
|               | Lines (Non-test)   | 15,163.38| 13,777   | 1,039,288 |
| Gold patch    | Files edited       | 2.6      | 3        | 7         |
|               | Lines edited       | 56       | 76       | 300       |
| Tests         | Fail to Pass       | 10.94    | 5        | 4,941     |
|               | Total              | 58.5     | 49       | 7,820     |

# Dataset Structure
The dataset contains the following fields. It includes all fields from SWE-bench and adds a `meta` column, which indicates whether the instance meets the "lite" criteria and, if not, lists the failed validators.

| Field name                 | Type   | Description                                                                                     |
|----------------------------|--------|-------------------------------------------------------------------------------------------------|
| `instance_id`              | str    | A formatted instance identifier, usually as `repo_owner__repo_name-PR-number`.                 |
| `patch`                    | str    | The gold patch, the patch generated by the PR (minus test-related code), that resolved the issue. |
| `repo`                     | str    | The repository owner/name identifier from GitHub.                                              |
| `base_commit`              | str    | The commit hash of the repository representing the HEAD of the repository before the solution PR is applied. |
| `hints_text`               | str    | Comments made on the issue prior to the creation of the solution PR’s first commit creation date. |
| `created_at`               | str    | The creation date of the pull request.                                                         |
| `test_patch`               | str    | A test-file patch that was contributed by the solution PR.                                     |
| `problem_statement`        | str    | The issue title and body.                                                                      |
| `version`                  | str    | Installation version to use for running evaluation.                                            |
| `environment_setup_commit` | str    | Commit hash to use for environment setup and installation.                                     |
| `FAIL_TO_PASS`             | str    | A JSON list of strings that represent the set of tests resolved by the PR and tied to the issue resolution. |
| `PASS_TO_PASS`             | str    | A JSON list of strings that represent tests that should pass before and after the PR application. |
| `meta`                     | str    | A JSON dictionary indicating whether the instance is lite, along with a list of failed lite validators if it is not. |
| `license`                  | str    | The type of license of the repository. |

To execute instances within SWE-bench, you need to provide a default recipe for dependency installation. The constants required for running these instances are described in this [constants.py](https://huggingface.co/datasets/nebius/SWE-bench-extra/blob/main/constants.py).

# License
The dataset is licensed under the Creative Commons Attribution 4.0 license. However, please respect the license of each specific repository on which a particular instance is based. To facilitate this, the license of each repository at the time of the commit is provided for every instance.
>>>>>>> 800c97d (data)
