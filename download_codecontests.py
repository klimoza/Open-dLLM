import numpy as np
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
 
codecontest = load_dataset("deepmind/code_contests")
 
PYTHON_ID = 1
 
 
def has_python(row):
    language_ids = row['solutions']['language']
    has_python = PYTHON_ID in language_ids
    return has_python
 
 
pythoncontest = codecontest.filter(has_python)
pythoncontest_easy = codecontest.filter(lambda row: row['difficulty'] <= 3)
 
 
def transform_test_list(row):
    row['public_tests'] = [{'input': x, 'output': y} for x, y in
                           zip(row['public_tests']['input'], row['public_tests']['output'])]
    row['generated_tests'] = [{'input': x, 'output': y} for x, y in
                              zip(row['generated_tests']['input'], row['generated_tests']['output'])]
    return row
 
 
pythoncontest_easy = pythoncontest_easy.map(transform_test_list)
 
pythoncontest_easy = pythoncontest_easy.filter(lambda row: len(row['generated_tests']) >= 5)
 
tokenizer = AutoTokenizer.from_pretrained("fredzzp/open-dcoder-0.5B")
 
 
def get_prompt(row):
    messages = [
        {"role": "system",
         "content": "You are a coding assistant.  Your task is to output ONLY valid code for the given task.  Do not include explanations, comments, markdown formatting (```), or natural language.  Output exactly and only the code required. Use input() to input and print() to output."},
        {"role": "user", "content": "Solve the following problem:\n\n" + row['description']}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
 
    row['prompt'] = text
    return row
 
 
pythoncontest_easy = pythoncontest_easy.map(get_prompt)
print(pythoncontest_easy.shape)
pythoncontest_easy = pythoncontest_easy.filter(lambda x: len(tokenizer.tokenize(x['prompt'])) <= 512)
print(pythoncontest_easy.shape)
 
# Combine all splits into a single list of examples
all_examples = []
for split in pythoncontest_easy.keys():
    all_examples.extend(pythoncontest_easy[split])
 
# Shuffle the combined examples
rng = np.random.default_rng(42)
indices = np.arange(len(all_examples))
rng.shuffle(indices)
all_examples = [all_examples[i] for i in indices]
 
# Define new split sizes (e.g., 90% train, 10% test)
n_total = len(all_examples)
n_test = max(1, int(0.1 * n_total))
n_train = n_total - n_test
 
train_examples = all_examples[:n_train]
test_examples = all_examples[n_train:]
val_examples = test_examples[:50]
breakpoint()
# Create new DatasetDict with train and test splits
codecontests = DatasetDict({
    "train": Dataset.from_list(train_examples),
    "test": Dataset.from_list(test_examples),
    "val": Dataset.from_list(val_examples)
})
 
codecontests.save_to_disk('datasets/codecontests')