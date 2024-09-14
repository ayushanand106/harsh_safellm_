import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    LxmertTokenizer,
    LxmertForQuestionAnswering,
    Blip2Processor,
    Blip2Model,
    TrainingArguments,
    Trainer,
)
from transformers.utils import logging

logging.set_verbosity_info()

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LXMERT tokenizer and model
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased").to(device)

# Load BLIP-2 model and processor for image feature extraction
blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").to(device)
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")

def extract_image_features(image):
    inputs = blip2_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = blip2_model.get_image_features(**inputs)
    return image_features  # Shape: (batch_size, seq_len, feature_dim)

# Custom Dataset Class
class VQADataset(Dataset):
    def __init__(self, data_dict, image_dir, tokenizer, answer_dict):
        self.image_filenames = list(data_dict.keys())
        self.data_dict = data_dict
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.answer_dict = answer_dict

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        input_text, target_text = self.data_dict[image_filename]
        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path).convert("RGB")

        # Extract image features using BLIP-2
        visual_feats = extract_image_features(image)  
        # print(visual_feats)

        # Adjust visual_feats to match LXMERT expected input
        visual_feats = visual_feats['last_hidden_state'].squeeze(0)  # Shape: (seq_len, feature_dim)
        num_features = visual_feats.size(0)

        # Create dummy visual_pos (since we don't have actual bounding boxes)
        visual_pos = torch.zeros(num_features, 4).to(device)  # Shape: (num_features, 4)

        # Tokenize input text
        question_inputs = self.tokenizer(
            input_text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Prepare labels
        answer = target_text.lower()
        if answer in self.answer_dict:
            label = torch.tensor(self.answer_dict[answer], dtype=torch.long).to(device)
        else:
            # If the answer is not in the answer_dict, label it as -1 (ignored)
            label = torch.tensor(-1, dtype=torch.long).to(device)

        # Prepare the final input dict
        inputs = {
            "input_ids": question_inputs["input_ids"].squeeze(0),
            "attention_mask": question_inputs["attention_mask"].squeeze(0),
            "visual_feats": visual_feats,
            "visual_pos": visual_pos,
            "labels": label,
        }

        return inputs

# Load data from JSON file
data_json_path = "/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/download_status_train.json"  # Update with your JSON file path
image_dir = "/media/mlr_lab/325C37DE7879ABF2/AyushAnand/ml_challenge/training"          # Update with your image directory path

with open(data_json_path, "r") as f:
    data_dict = json.load(f)

# Build answer dictionary (answer to index mapping)
all_answers = [entry[1].lower() for entry in data_dict.values() if entry[1]!=None]
unique_answers = list(set(all_answers))
answer_dict = {answer: idx for idx, answer in enumerate(unique_answers)}
num_labels = len(answer_dict)
print(f"Number of unique answers: {num_labels}")

# Update model's classification head to match the number of labels
model.config.num_labels = num_labels
model.cls = torch.nn.Linear(model.config.hidden_size, num_labels).to(device)

# Create the dataset
dataset = VQADataset(data_dict, image_dir, tokenizer, answer_dict)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Custom data collator
def custom_data_collator(features):
    batch = {}
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features]).to(device)
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features]).to(device)
    batch["visual_feats"] = torch.stack([f["visual_feats"] for f in features]).to(device)
    batch["visual_pos"] = torch.stack([f["visual_pos"] for f in features]).to(device)
    batch["labels"] = torch.stack([f["labels"] for f in features]).to(device)
    return batch

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on your GPU memory
    per_device_eval_batch_size=2,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
)

# Define compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    # Filter out ignored labels (-1)
    valid_indices = labels != -1
    correct = (predictions[valid_indices] == labels[valid_indices]).sum()
    total = valid_indices.sum()
    accuracy = (correct / total).item() if total > 0 else 0.0
    return {"accuracy": accuracy}

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_data_collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
