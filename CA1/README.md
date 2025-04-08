# 🧠 Emotion Detection with Large Language Models

## 📌 Overview

This project explores the use of Large Language Models (LLMs) for emotion classification. We evaluate and compare the performance of three models:

- **Base Model:** `meta-llama/Llama-3.2-1B`
- **Instruction-Tuned Model:** `meta-llama/Llama-3.2-1B-Instruct`
- **LoRA Fine-Tuned Model:** Fine-tuned on an emotion-labeled dataset using LoRA (Low-Rank Adaptation)

The models are tested on their ability to classify emotions using standard evaluation metrics, demonstrating the strengths of instruction tuning and parameter-efficient fine-tuning.

## 🎯 Key Learnings

- Formatting and preprocessing data for classification with LLMs
- Applying LoRA for efficient fine-tuning
- Evaluating zero-shot, instruction-based, and fine-tuned models
- Computing and comparing Accuracy and F1-Score
- Crafting effective prompts for classification tasks

## 📚 Dataset

The dataset includes text samples labeled with one of six emotions:

- **joy**
- **sadness**
- **love**
- **anger**
- **fear**
- **surprise**

### Sample Format (JSON)
```json
{
  "text": "I miss you so much.",
  "label": "sadness"
}
```

This data is used to fine-tune the LLM and evaluate model performance.

## ⚙️ Methods

We evaluate three model types:

- **Base Model:**  
  `meta-llama/Llama-3.2-1B`  
  Used without tuning for zero-shot classification.

- **Instruction-Tuned Model:**  
  `meta-llama/Llama-3.2-1B-Instruct`  
  Evaluated on its ability to follow task-specific prompts.

- **LoRA Fine-Tuned Model:**  
  Fine-tuned on the emotion classification dataset using LoRA, a parameter-efficient method that modifies only a small subset of weights.

## 🧪 Evaluation Metrics

- **Accuracy:** Proportion of correct predictions
- **F1-Score:** Harmonic mean of precision and recall, valuable for handling class imbalances

---