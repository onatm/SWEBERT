import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import numpy as np
import evaluate


MODEL_PATH = "./SWEBERT"
DATA_PATH = "./data/training_data.csv"


def prepare_data():
    """Prepares the dataset for training and evaluation by loading from CSV."""
    try:
        # Load data from CSV file using datasets library
        # The file path needs to be a dictionary with a key that represents the split name
        dataset = load_dataset("csv", data_files={"train": DATA_PATH})

        # Extract unique labels
        labels = ["database", "networking", "machine-learning"]

        # Create label mappings
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}

        # Create a function to convert string labels to numerical IDs
        def convert_labels(examples):
            examples["label"] = [label2id[label] for label in examples["label"]]
            return examples

        # Apply the conversion to all examples in batch mode
        dataset = dataset.map(convert_labels, batched=True)

        # Split dataset into train and test
        split_dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)

        return split_dataset, len(labels), id2label, label2id
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print(
            f"Make sure the file {DATA_PATH} exists and is a valid CSV with 'text' and 'label' columns."
        )
        raise


def compute_metrics(eval_pred):
    """Computes accuracy metric for evaluation."""
    accuracy = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


def train_and_evaluate(dataset, num_labels, id2label, label2id):
    """Trains and evaluates the model."""
    # --- Step 2: Preprocessing (Tokenization) ---

    model_checkpoint = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], padding=True, truncation=True, return_tensors="pt"
        )

    if "label" in dataset["train"].features.keys():
        dataset = dataset.rename_column("label", "labels")

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # --- Step 3: Load the Pre-trained Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- Step 4: Define Training Arguments & Train ---
    training_args = TrainingArguments(
        output_dir="./checkpoints",
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\n--- Starting ModernBERT Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # --- Step 5: Evaluate the Model ---
    print("\n--- Evaluating Model ---")
    eval_results = trainer.evaluate()
    print(f"Evaluation Accuracy: {eval_results['eval_accuracy']:.4f}")

    # --- Step 6: Save Model ---
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def classify_articles():
    """Uses the fine-tuned model to classify new articles."""
    classifier = pipeline("text-classification", model=MODEL_PATH)

    print("\n--- Testing Model with New Articles ---")

    new_article = (
        "This article discusses TCP/IP and HTTP protocols for fast data transfer."
    )
    result = classifier(new_article)
    print(f"Article: '{new_article}'")
    print(f"Prediction: {result[0]['label']} (Score: {result[0]['score']:.4f})\n")

    new_article_2 = "The data was stored in a PostgreSQL database."
    result_2 = classifier(new_article_2)
    print(f"Article: '{new_article_2}'")
    print(f"Prediction: {result_2[0]['label']} (Score: {result_2[0]['score']:.4f})\n")

    new_article_3 = "A new SQL injection vulnerability was discovered."
    result_3 = classifier(new_article_3)
    print(f"Article: '{new_article_3}'")
    print(f"Prediction: {result_3[0]['label']} (Score: {result_3[0]['score']:.4f})\n")


def main():
    """Main function to run the text classification pipeline."""
    dataset, num_labels, id2label, label2id = prepare_data()
    train_and_evaluate(dataset, num_labels, id2label, label2id)
    classify_articles()


if __name__ == "__main__":
    main()
