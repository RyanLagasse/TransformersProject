import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess data
df = pd.read_csv('spam.csv')
texts = df['text'].tolist()
labels = df['label'].map({'spam': 1, 'not_spam': 0}).tolist()

# Split into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# Initialize the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize the texts
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# Create the PyTorch datasets
train_dataset = list(zip(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels))
test_dataset = list(zip(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels))

# Load the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
model.save_pretrained('./spam_detector')