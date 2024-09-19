from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, Dataset
import pandas as pd

# Load the model and tokenizer
model_name = "predictivemodel/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Load the dataset
dataset = load_dataset("emotion", split="train")

# Convert the dataset to a pandas DataFrame
df = pd.DataFrame(dataset)

# Save the DataFrame to a CSV file
csv_file_path = "emotion_dataset_train.csv"
df.to_csv(csv_file_path, index=False)

print(f"Dataset exported to {csv_file_path}")
