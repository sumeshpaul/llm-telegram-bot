from datasets import load_dataset

# Load and preview the dataset
dataset = load_dataset("json", data_files="train_dataset_v2.json")

# Optional: split if needed later
dataset = dataset["train"].train_test_split(test_size=0.1)

print(f"✅ Loaded dataset with {len(dataset['train'])} train and {len(dataset['test'])} test examples.")
print(dataset)
