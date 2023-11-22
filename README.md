# Training and Evaluating LLMs on non-English languages

At Abzu, we frequently explore the idea of expanding our chat agents’ capabilities to include multiple languages. The critical questions we face include the cost of such an expansion and whether the model would retain its logical skills. Our aim is to delve into these aspects by examining and contrasting the Falcon and Aguila models.

Some of the projects used to do these experiments are:

Name: Ǎguila-7B
Source: https://github.com/projecte-aina/lm-catalan
Description: Ǎguila-7B is a 7B parameters LLM that has been trained on a mixture of Spanish, Catalan, and English data, adding up to a total of 26B tokens. It uses the Falcon-7b model as a starting poin.

Name: mt-aina-ca-en
Source: https://huggingface.co/projecte-aina/mt-aina-ca-en
Description: The mt-aina-ca-en model is a machine translation model that was trained from scratch using the Fairseq toolkit on a combination of Catalan-English datasets, amounting to 11 million sentences.

Name: falcon-7b
Source: https://huggingface.co/tiiuae/falcon-7b
Description: Falcon-7B is a 7B parameters causal decoder-only model built by the Technology Innovation Institute (TII) and trained on 1,500B tokens of RefinedWeb enhanced with curated corpora.

# How to reproduce the results

Clone the repository and install requirements in a clean venv
```bash
> python -m venv venv
> source ./venv/bin/activate
> pip install -r requirements.txt
```

Edit evals_f1.py and select the model to run by editing the import line.

```python
...
model_id =  "tiiuae/falcon-7b" # "projecte-aina/aguila-7b"
...
```

Run the evals script

```bash
> python evals_f1.py
```

Run the scafalcon results

```bash
> python scafalcon.py
```

# Look at the predictions

You can also look at some of the results in the results.ipynb notebook and run some examples.
