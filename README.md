# Training and Evaluating LLMs on non-English languages

At Abzu, we frequently explore the idea of expanding our chat agents’ capabilities to include multiple languages. The critical questions we face include the cost of such an expansion and whether the model would retain its logical skills. Our aim is to delve into these aspects by examining and contrasting the Falcon and Aguila models.

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
