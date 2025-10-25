"""
Fine-tuning of AST
"""

import os
import numpy as np

import torch, evaluate
import argparse

from transformers import ASTConfig, ASTForAudioClassification
from transformers import ASTFeatureExtractor
from transformers import Trainer, TrainingArguments

# pre-trained model name
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"

parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu")
args = parser.parse_args()

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class fine_tune_AST():
    def __init__(self, categories, model=None, mean=0, std=1):
        # load configuration for pre-trained AST model
        self.config = ASTConfig.from_pretrained(pretrained_model)

        # config update: id2label as per mapping
        self.config.num_labels = len(categories)
        self.config.id2label = {i: category["name"] for i, category in enumerate(categories)}
        self.config.label2id = {v: k for k, v in self.config.id2label.items()}

        self.feature_extractor = ASTFeatureExtractor.from_pretrained(pretrained_model)
        self.feature_extractor.mean = mean
        self.feature_extractor.std = std
        self.feature_extractor.do_normalize = True

        # we save model input name and sampling rate for later use
        self.model_input_name = self.feature_extractor.model_input_names[0]  # key -> 'input_values'
        self.SAMPLING_RATE = self.feature_extractor.sampling_rate

        if model is not None:
            self.model = model
        else:
            self.model = ASTForAudioClassification.from_pretrained(pretrained_model,
                                                                   config=self.config,
                                                                   ignore_mismatched_sizes=True)
        self.model.init_weights()

    def preprocess_audio(self, batch_data):
        wavs = [audio["array"] for audio in batch_data["input_values"]]
        # inputs are spectrograms as torch.tensors now
        inputs = self.feature_extractor(wavs, sampling_rate=self.SAMPLING_RATE, return_tensors="pt")

        output_batch = {self.model_input_name: inputs.get(self.model_input_name), "labels": list(batch_data["labels"])}
        return output_batch

    def compute_metrics(self, eval_pred):
        accuracy = evaluate.load("accuracy")
        recall = evaluate.load("recall")
        precision = evaluate.load("precision")
        f1 = evaluate.load("f1")

        AVERAGE = "macro" if self.config.num_labels > 2 else "binary"

        logits = eval_pred.predictions
        predictions = np.argmax(logits, axis=1)
        metrics = accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
        metrics.update(precision.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(recall.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        metrics.update(f1.compute(predictions=predictions, references=eval_pred.label_ids, average=AVERAGE))
        return metrics
    
    def training(self, dataset, lr=5e-5, N_epochs=10, batch_size=8, output_dir='./trained_ast_clf'):
        # Configure training run with TrainingArguments class
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=lr,  # Learning rate
            push_to_hub=False,
            num_train_epochs=N_epochs,  # Number of epochs
            per_device_train_batch_size=batch_size,  # Batch size per device
            eval_strategy="epoch",  # Evaluation strategy
            # save_strategy="epoch",
            eval_steps=1,
            # save_steps=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            # logging_strategy="steps",
            # logging_steps=20,
        )

        # Setup the trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=self.compute_metrics,  # Use the metrics function from above
        )

        # training run
        trainer.train()

