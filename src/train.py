# Colab File

import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Load data
df = pd.read_csv("data/zends_dataset_v2.csv")

# Label encode
# I converted categorical intent labels into numerical format using LabelEncoder.
le = LabelEncoder()
df['intent_label'] = le.fit_transform(df['intent'])

# Train-Test Split
# I split the data into training and validation sets to evaluate model performance.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'], df['intent_label'], test_size=0.2, random_state=42
)

# Tokenize
# I tokenized the text using a pretrained DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

# Dataset class
# I created a custom PyTorch dataset class to feed tokenized data into the model.
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Dataset Objects(ready for model)
train_dataset = IntentDataset(train_encodings, list(train_labels))
val_dataset = IntentDataset(val_encodings, list(val_labels))

# Load model
# I used DistilBERT for sequence classification with the number of labels equal to the number of intents.
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# Training Arguments
# I configured training parameters like epochs and batch size using TrainingArguments.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    report_to="none"
)

# I trained the model using HuggingFace Trainer API.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# ✅ Save model + encoder
model.save_pretrained("models/intent_model")
tokenizer.save_pretrained("models/intent_model")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model saved successfully!")
