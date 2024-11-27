# Problem: Build a simple transformer-based model (BERT) for text classification (IMDb Dataset)

from transformers import BertTokenizer, TFBertForSequenceClassification
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam

# Load the data
# This dataset contains 50,000 movie reviews,
# split equally into training and testing sets,
# with labels indicating whether the review is positive (1) or negative (0).
dataset = load_dataset('imdb')

# Load the Bert Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_func(movie):
    return tokenizer(movie['text'], padding='max_length', truncation=True, max_length=128)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_func, batched=True)

# Prepare the Dataset
# Convert the tokenized dataset into a TensorFlow-friendly format.

train_dataset = tokenized_dataset['train'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='label',
    shuffle=True,
    batch_size=16
)

test_dataset = tokenized_dataset['test'].to_tf_dataset(
    columns=['input_ids', 'attention_mask'],
    label_cols='label',
    shuffle=False,
    batch_size=16
)

# Build Bert model for classification
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=3
)

# Evaluate the Model
loss, accuracy = model.evaluate(test_dataset)
print(f"Loss is {loss} and accuracy is: {accuracy}")
