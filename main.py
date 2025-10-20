import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from sklearn.metrics import f1_score


MODEL_PATH = "./SWEBERT"
DATA_PATH = "./data/training_data.csv"


def prepare_data():
    """Prepares the dataset for training and evaluation by loading from CSV."""
    # Load data from CSV file using datasets library
    # The file path needs to be a dictionary with a key that represents the split name
    dataset = load_dataset("csv", data_files={"train": DATA_PATH}, delimiter=";")

    # Extract unique labels
    labels = ["database", "networking", "machine-learning"]

    # Create label mappings
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    # Create a function to convert comma-separated string labels to multi-hot vectors
    def convert_labels(examples):
        multi_labels = []
        for lab_str in examples["label"]:
            label_items = [item.strip() for item in lab_str.split(",") if item.strip()]
            vector = [1.0 if lbl in label_items else 0.0 for lbl in labels]
            multi_labels.append(vector)
        examples["labels"] = multi_labels
        return examples

    # Apply the conversion and drop the original 'label' column
    dataset = dataset.map(convert_labels, batched=True, remove_columns=["label"])

    # Split dataset into train and test
    split_dataset = dataset["train"].train_test_split(test_size=0.3, seed=42)  # type: ignore[attr-defined]

    return split_dataset, len(labels), id2label, label2id


def compute_metrics(eval_pred):
    """Computes accuracy metric for evaluation."""
    # Compute micro F1 using sklearn for multi-label
    logits, labels = eval_pred
    # apply sigmoid and threshold
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    refs = labels.astype(int)
    score = f1_score(refs, preds, average="micro")
    return {"micro_f1": score}


# def compute_metrics(eval_pred):
#     """Computes accuracy metric for evaluation."""
#     accuracy = evaluate.load("accuracy")
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels) # type: ignore


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
    model.config.problem_type = "multi_label_classification"

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
    # Print multi-label F1 instead of accuracy
    print(f"Evaluation micro F1: {eval_results['eval_micro_f1']:.4f}")

    # --- Step 6: Save Model ---
    trainer.save_model(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def classify_articles():
    """Uses the fine-tuned model to classify new articles."""
    # load tokenizer and use multi-label pipeline
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    classifier = pipeline(
        "text-classification",
        model=MODEL_PATH,
        tokenizer=tokenizer,
        function_to_apply="sigmoid",
        return_all_scores=True,
    )

    print("\n--- Testing Model with New Articles ---")

    new_article = (
        "This article discusses TCP/IP and HTTP protocols for fast data transfer."
    )
    # classify single article by wrapping in list and taking first result
    # classify in batch form to get list of lists
    scores = classifier([new_article])
    scores = scores[0]  # get predictions for first (and only) input
    preds = [r for r in scores if isinstance(r, dict) and r["score"] > 0.5]
    print(f"Article: '{new_article}'")
    print(f"Predictions: {preds}\n")

    new_article_2 = "The data was stored in a PostgreSQL database."
    scores2 = classifier([new_article_2])[0]
    preds2 = [r for r in scores2 if isinstance(r, dict) and r["score"] > 0.5]
    print(f"Article: '{new_article_2}'")
    print(f"Predictions: {preds2}\n")

    new_article_3 = "A new SQL injection vulnerability was discovered that uses tcp/ip stack."
    scores3 = classifier([new_article_3])[0]
    preds3 = [r for r in scores3 if isinstance(r, dict) and r["score"] > 0.5]
    print(f"Article: '{new_article_3}'")
    print(f"Predictions: {preds3}\n")


def main():
    """Main function to run the text classification pipeline."""
    dataset, num_labels, id2label, label2id = prepare_data()
    train_and_evaluate(dataset, num_labels, id2label, label2id)
    classify_articles()


if __name__ == "__main__":
    main()
