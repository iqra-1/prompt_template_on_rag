# Prompt Template Evaluation on Retrieval-Augmented Generation (RAG)

## Abstract
Retrieval-Augmented Generation (RAG) enhances large language models by combining external knowledge retrieval with neural text generation. While RAG has shown strong potential for knowledge-intensive tasks, the effect of **prompt template design** on its performance is not well understood.

In this study, we evaluated multiple prompt templates on Facebook’s official `facebook/rag-token-nq` architecture, using authentic Wikipedia data (21 GB compressed index, 75 GB RAM requirement). We ran **four experiments** across the SQuAD 2.0, NQ-Open, and TriviaQA datasets, performing over **11,500 evaluations** and measuring performance via F1 score.

Our results show that **prompt engineering can significantly influence RAG performance**, with impacts ranging from **–37.4 % to +9.7 %**. Notably, small and simple changes often yielded the best results — for example, adding a single question mark improved performance by 9.7 % (F1 = 0.118), and the basic template achieved the **highest overall F1 score (0.223)** in multi-dataset evaluations. In contrast, templates with expert roles, lengthy instructions, or synthesis directives consistently underperformed.

These findings challenge the assumption that more complex prompts lead to better results, suggesting that **syntactic clarity and minimalism** may be more effective for RAG systems. The experiments provide actionable guidelines for practitioners: careful template selection can yield substantial improvements without changing the architecture or increasing computational resources.

**Keywords:** Retrieval-Augmented Generation, Prompt Engineering, NLP, Question Answering, Large Language Models, Template Design

---

## Repository Structure
```

prompt+rag/
├── scripts/            # Python scripts for experiments
├── results/            # Output CSV files from experiments
├── requirements.txt
└── README.md

````

---

## Installation
1. Clone the repository:
```bash
git clone https://github.com/iqra-1/prompt_template_on_rag.git
cd prompt_template_on_rag
````

2. (Optional) Create a conda environment:

```bash
conda create -n rag_env python=3.10
conda activate rag_env
pip install -r requirements.txt
```

---

## Running Experiments

The experiment scripts will **download the Wikipedia embedding index (\~21 GB compressed, requires \~75 GB RAM)** using HuggingFace datasets/cache. Make sure you have enough space before running.

1. Navigate to the `scripts` folder:

```bash
cd scripts
```

2. Run the experiments:

```bash
python exp-1.py   # Run experiment 1
python exp-2.py   # Run experiment 2
python exp-3.py   # Run experiment 3
python exp-4.py   # Run experiment 4
```

> ⚠️ Note: The Wikipedia embedding index will be downloaded in the HuggingFace cache (usually `~/.cache/huggingface/`).

---

## Results

Experiment results will be saved as `.csv` files in the `results/` folder. The files include metrics such as F1 scores and other evaluation statistics.

---
