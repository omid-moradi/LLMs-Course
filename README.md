# Large Language Models (LLMs) Course Projects - University of Tehran (Spring 2025)

This repository contains projects completed as part of the **Large Language Models (LLMs)** course at the University of Tehran (Spring 2025). The projects cover a range of topics related to LLMs, including word embeddings, fine-tuning, retrieval-augmented generation, reinforcement learning from human feedback (Alignment), and more.

## Projects

Below is a detailed description of each Jupyter notebook in this repository:

* **`LLM_CA1.ipynb`**:
    This notebook serves as an introduction to working with Large Language Models. It covers fundamental concepts such as:
    * **Getting Started with LLMs**: Basic setup and interaction with LLMs.
    * **Tokenization**: Understanding how text is broken down into tokens, comparing different tokenizers (LLaMA, Mistral, Phi), and decoding model generations.
    * **Model Comparison**: Analyzing the differences in output between base models and instruction-tuned models (e.g., Llama-3.2-1B vs. Llama-3.2-1B-Instruct).
    * **Chat Templates**: Exploring and applying chat templates for instruction-tuned models to manage multi-turn conversations.
    * **Fine-tuning using LoRA**: An extensive section on fine-tuning a base model for emotion detection using LoRA (Low-Rank Adaptation), including dataset preparation (emotion dataset), experimenting with LoRA configurations, and understanding memory usage in fine-tuning.

* **`LLM_CA2.ipynb`**:
    This notebook focuses on **In-context Learning (ICL)** and various **Prompt Engineering** strategies for Large Language Models. Key areas include:
    * **Introduction to In-context Learning**: Explaining what ICL is, its differences from fine-tuning, and its limitations.
    * **Chain-of-Thought (CoT) Prompting**: Detailed explanation of CoT and its mechanics.
    * **Model Loading**: Utilizing the `unsloth` library to efficiently load `Llama-3.2-3B-Instruct-bnb-4bit` for inference.
    * **Benchmark Dataset**: Loading and sampling from the `GSM8K` benchmark for evaluation.
    * **Prompt Engineering Strategies**: Implementation and evaluation of multiple prompting techniques such as Zero-shot, Role-play, Zero-shot CoT, Few-shot CoT, Least-to-Most prompting, and Generated Knowledge prompting.
    * **Evaluation**: Assessing the performance of different prompting methods.

* **`LLM_HW3_Part1_LLM_as_a_Judge.ipynb`**:
    This notebook is the first part of Homework 3, delving into **Judgement Strategies in LLM as a Judge**. It guides users through:
    * **Dataset Exploration**: Loading, summarizing, and statistically analyzing the `prometheus-eval/Feedback-Bench` dataset, which is used for evaluating LLM feedback and alignment.
    * **Data Analysis**: Describing column representations, identifying numerical values, and plotting their distributions.
    * **LangChain Overview**: Introduces fundamental LangChain concepts, including loading models (`microsoft/Phi-4-mini-instruct`) and understanding key generation parameters (temperature, max_new_tokens, top_p, top_k, repetition_penalty).
    * **Simple Chains**: Building basic conversational chains using `HumanMessagePromptTemplate`, `AIMessagePromptTemplate`, and `StrOutputParser`.
    * **JSON Chains**: Creating more advanced chains to extract structured information (e.g., football player details) into JSON format using `SystemMessagePromptTemplate` and `JsonOutputParser`.

* **`LLM_HW3_Part2_RAG.ipynb`**:
    This notebook is the second part of Homework 3, dedicated to implementing and evaluating **Retrieval-Augmented Generation (RAG) pipelines**. It covers:
    * **Information Retrieval (IR) and RAG Overview**: Explains the concepts of IR and RAG, their architectures, and the benefits of RAG over traditional generative models.
    * **LangChain Framework for RAG**: Demonstrates how LangChain facilitates building RAG pipelines.
    * **Dataset Preparation**: Loading and processing the `RecipeNLG` dataset (5000 recipes), including converting list objects (ingredients, directions, NER) into strings and splitting documents into manageable chunks using `RecursiveCharacterTextSplitter`.
    * **Sparse Retriever**: Implementation of a TF-IDF-based sparse retriever using `TFIDFRetriever` for keyword-based document retrieval.
    * **Semantic Retriever**: Creation of a semantic retriever utilizing `BAAI/bge-small-en` as the embedding model and `FAISS` as the vector store for semantic similarity-based retrieval.
    * **RAG Pipeline Construction**: Building end-to-end RAG pipelines incorporating both sparse and semantic retrievers.
    * **Pipeline Evaluation**: Designing and executing queries to compare the performance of RAG pipelines (sparse and semantic) against a standalone LLM using qualitative analysis of responses.

* **`CA4_Part1_Quantization_Self_Explanations.ipynb`**:
    This notebook focuses on **Quantization** and **Self-Explanations** in Large Language Models. It provides practical insights into:
    * **Quantization Fundamentals**: Explaining the concept of quantization as a technique to reduce model precision (e.g., from 32-bit floating-point to 8-bit or 4-bit integers) to minimize memory footprint and speed up inference.
    * **Simple Quantization Example**: Demonstrating quantization and dequantization with a simple numerical example and illustrating precision loss by plotting a function before and after quantization.
    * **4-bit Quantization and QLoRA**: Discussing 4-bit quantization and its implications for efficient LLM deployment.
    * **Self-Explanations**: Covering techniques for models to generate their own explanations for their outputs.
