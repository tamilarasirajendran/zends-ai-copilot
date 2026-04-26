# Train a DistilBERT model to classify user queries into intents
# I fine-tuned a DistilBERT model for intent classification using a labeled dataset.

import pandas as pd
import torch
import pickle
from sklearn.model_selection import train_test_split #Split data
from sklearn.preprocessing import LabelEncoder # convert labels
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
) #Load tokenizer, model, trainer

# LOAD DATASET
# I used a synthetic dataset generated earlier for training.

df = pd.read_csv("data/zends_dataset_v2.csv")   

# LABEL ENCODING
# I encoded categorical intent labels into numerical form for model training.

le = LabelEncoder()
df['intent_label'] = le.fit_transform(df['intent']) #Convert text → numbers


#  TRAIN-TEST SPLIT
# I split the dataset into training and validation sets while maintaining class distribution.
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'],
    df['intent_label'],
    test_size=0.2,
    random_state=42,
    stratify=df['intent_label'] #keeps balance
)

# TOKENIZATION
# I tokenized the input text using a pretrained DistilBERT tokenizer.
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") #Convert text → tokens

train_encodings = tokenizer(
    list(train_texts),
    truncation=True,
    padding="max_length",
    max_length=128 #Controls input size
)

val_encodings = tokenizer(
    list(val_texts),
    truncation=True,
    padding="max_length",
    max_length=128
)


# DATASET CLASS
# I created a custom dataset class to feed tokenized data into the model.
class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# DATASET OBJECTS

train_dataset = IntentDataset(train_encodings, train_labels)
val_dataset = IntentDataset(val_encodings, val_labels)

# LOAD MODEL
# I used DistilBERT for sequence classification with multiple intent labels.
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
) #Load DistilBERT

# METRICS
# I evaluated the model using accuracy and F1 score.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# TRAINING ARGUMENTS
# I configured training parameters like epochs, batch size, and evaluation strategy.
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,  # faster for demo
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none"
)

# TRAINER
# I used Hugging Face Trainer API to simplify training and evaluation.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# TRAIN

trainer.train() #Model learns patterns

# SAVE MODEL
model.save_pretrained("models/intent_model")
tokenizer.save_pretrained("models/intent_model")

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model trained and saved successfully!")