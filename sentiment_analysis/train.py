<<<<<<< HEAD
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset, concatenate_datasets

TRAIN_FILE_PATH = "datasets/twitter_dataset/sent_train.csv"          # Twitter Financial News train dataset
VALIDATION_FILE_PATH = "datasets/twitter_dataset/sent_valid.csv" # Twitter Financial News validation dataset
HEADLINES_FILE_PATH = "datasets/headline_dataset/all-data.csv"    # Financial News Headline dataset

label_map = {
    "LABEL_0": 0,  # Bearish
    "LABEL_1": 1,  # Bullish
    "LABEL_2": 2   # Neutral
}

data_files = {"train": TRAIN_FILE_PATH, "validation": VALIDATION_FILE_PATH}
twitter_dataset = load_dataset('csv', data_files=data_files)

def encode_twitter_labels(example):
    example["label"] = label_map[example["sentiment"]]
    return example

twitter_dataset = twitter_dataset.map(encode_twitter_labels)

def load_headlines_dataset(file_path):
    headlines = load_dataset('csv', data_files={"train": file_path}, delimiter=";")
    headlines = headlines['train'].train_test_split(test_size=0.2)
    return headlines

headlines_dataset = load_headlines_dataset(HEADLINES_FILE_PATH)

def encode_headlines_labels(example):
    if example["Sentiment"] == "negative":
        example["label"] = 0  # Bearish
    elif example["Sentiment"] == "positive":
        example["label"] = 1  # Bullish
    elif example["Sentiment"] == "neutral":
        example["label"] = 2  # Neutral
    return example

headlines_dataset = headlines_dataset.map(encode_headlines_labels)

# Concatenate
combined_train = concatenate_datasets([twitter_dataset["train"], headlines_dataset["train"]])
combined_validation = concatenate_datasets([twitter_dataset["validation"], headlines_dataset["test"]])

model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize
def tokenize_function(example):
    return tokenizer(example["text"] if "text" in example else example["News Headline"],
                     truncation=True, padding="max_length", max_length=128)

tokenized_train = combined_train.map(tokenize_function, batched=True)
tokenized_validation = combined_validation.map(tokenize_function, batched=True)

tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_validation = tokenized_validation.rename_column("label", "labels")
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_validation.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

training_args = TrainingArguments(
    output_dir="./financial_bert_sentiment_combined",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Starting training on combined dataset...")
trainer.train()

print("Evaluating on validation set...")
eval_result = trainer.evaluate()
print(f"Validation Results: {eval_result}")

trainer.save_model("fine_tuned_financial_bert_combined")
tokenizer.save_pretrained("fine_tuned_financial_bert_combined")

print("Model saved in 'fine_tuned_financial_bert_combined' directory.")
=======
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, concatenate_datasets

# File paths
TRAIN_FILE_PATH = "datasets/twitter_dataset/sent_train.csv"
VALIDATION_FILE_PATH = "datasets/twitter_dataset/sent_valid.csv"
HEADLINES_FILE_PATH = "datasets/headline_dataset/all-data.csv"

# Load Twitter dataset
data_files = {"train": TRAIN_FILE_PATH, "validation": VALIDATION_FILE_PATH}
twitter_dataset = load_dataset('csv', data_files=data_files)

# Load Financial Headlines dataset
def load_headlines_dataset(file_path):
    try:
        headlines = load_dataset(
            'csv',
            data_files={"train": file_path},
            delimiter=",",
            encoding="ISO-8859-1"
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    # Map text and sentiment columns
    headlines = headlines.map(lambda example: {
        "text": example.get("text", example.get("News Headline", "")),
        "label": {"neutral": 2, "positive": 1, "negative": 0}.get(example.get("Sentiment", "neutral"), 2)
    })
    headlines = headlines['train'].train_test_split(test_size=0.2)
    return headlines

headlines_dataset = load_headlines_dataset(HEADLINES_FILE_PATH)

# Concatenate datasets
combined_train = concatenate_datasets([twitter_dataset["train"], headlines_dataset["train"]])
combined_validation = concatenate_datasets([twitter_dataset["validation"], headlines_dataset["test"]])

# Load model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenization
def tokenize_function(example):
    text = example.get("text", "")
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=100  # Match the expected length
    )

# Apply tokenization
tokenized_train = combined_train.map(tokenize_function, batched=True, batch_size=100)
tokenized_validation = combined_validation.map(tokenize_function, batched=True, batch_size=100)

# Prepare datasets for training
tokenized_train = tokenized_train.rename_column("label", "labels")
tokenized_validation = tokenized_validation.rename_column("label", "labels")
tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_validation.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Training arguments
training_args = TrainingArguments(
    output_dir="./financial_bert_sentiment_combined",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    logging_steps=10
)

# Metrics computation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Training
print("Starting training on combined dataset...")
trainer.train()

# Evaluation
print("Evaluating on validation set...")
eval_result = trainer.evaluate()
print(f"Validation Results: {eval_result}")

# Save model and tokenizer
trainer.save_model("fine_tuned_financial_bert_combined")
tokenizer.save_pretrained("fine_tuned_financial_bert_combined")
print("Model saved in 'fine_tuned_financial_bert_combined' directory.")
>>>>>>> f942399 (implemented sentiment analysis)
