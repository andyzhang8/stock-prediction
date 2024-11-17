from datasets import load_dataset

twitter_dataset = load_dataset("your_dataset_name", split="train")
print(twitter_dataset.column_names)
