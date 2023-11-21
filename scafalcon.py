# %%
import os
os.environ['XDG_CACHE'] = '/workspace/.cache'
os.environ['HF_HOME']='/workspace/.cache/huggingface'


# %%
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pyonmttok
import ctranslate2
from metrics import *
from huggingface_hub import snapshot_download


# %%
model_id =  "tiiuae/falcon-7b"
model_name = "scafalcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             torch_dtype=torch.bfloat16,
                                             trust_remote_code=True,
                                             device_map="auto")
model.config.pad_token_id = tokenizer.eos_token_id

# %%
print("Loading translator Models...")
ca_en_model_folder = snapshot_download(repo_id="projecte-aina/mt-aina-ca-en", revision="main")
tokenizer_ca_en = pyonmttok.Tokenizer(
    mode="none", sp_model_path=ca_en_model_folder + "/spm.model"
)
ca_en_model = ctranslate2.Translator(ca_en_model_folder, device="cuda")
en_ca_model_folder = snapshot_download(repo_id="projecte-aina/mt-aina-en-ca", revision="main")
tokenizer_en_ca = pyonmttok.Tokenizer(
    mode="none", sp_model_path=en_ca_model_folder + "/spm.model"
)
en_ca_model = ctranslate2.Translator(en_ca_model_folder, device="cuda")
# %%
def translate_to_english(txt):
    lines = [l for l in txt.split("\n") if l.strip() != ""]

    lines_chunks = []
    for line in lines:
        lines_chunks.append(line.split(". "))

    ts = []
    for i, line_chunk in enumerate(lines_chunks):
        toks, _ = tokenizer_ca_en.tokenize_batch(line_chunk)
        translated = ca_en_model.translate_batch(toks)
        ts.append([])
        for t in translated:
            t_str = tokenizer_ca_en.detokenize(t.hypotheses[0])
            ts[i].append(t_str)
        ts[i] = ". ".join(ts[i])

    return "\n".join(ts)

def translate_to_catalan(txt):
    lines = [l for l in txt.split("\n") if l.strip() != ""]

    lines_chunks = []
    for line in lines:
        lines_chunks.append(line.split(". "))

    ts = []
    for i, line_chunk in enumerate(lines_chunks):
        toks, _ = tokenizer_en_ca.tokenize_batch(line_chunk)
        translated = en_ca_model.translate_batch(toks)
        ts.append([])
        for t in translated:
            t_str = tokenizer_en_ca.detokenize(t.hypotheses[0])
            ts[i].append(t_str)
        ts[i] = ". ".join(ts[i])

    return "\n".join(ts)
# %%
def _run_llm(txt, num_tokens=20, stop_text='\n'):
    # Tokenize the input text
    keep_input = txt.split("\n")[-1]
    tokens = tokenizer(txt, return_tensors="pt").to(model.device)['input_ids']
    input_len = tokens.shape[1]

    # Calculate the total length of the output (input length + number of tokens to generate)
    if stop_text:
        stop_tokens = tokenizer(stop_text, return_tensors="pt").to(model.device)["input_ids"]
        stop_tokens_len = stop_tokens.shape[1]

    with torch.no_grad():
        # Generate tokens
        for _ in range(num_tokens):
            tokens = model.generate(tokens, do_sample=True, top_k=1, eos_token_id=tokenizer.eos_token_id, max_new_tokens=1)

            # If a stop text is found, truncate the output at its first occurrence
            if stop_text is not None:
                if (tokens[0][-stop_tokens_len:] == stop_tokens).all():
                    tokens[0][-stop_tokens_len:] = tokenizer.eos_token_id
                    break

        generated_only = tokenizer.decode(tokens[0][input_len:], skip_special_tokens=True)

        # Return from the last line
        return keep_input + generated_only

def run_inference(prompt, num_tokens=20, stop_text='\n'):
    prompt = translate_to_english(prompt)
    last_line = prompt.split("\n")[-1]
    last_line = translate_to_catalan(last_line)
    output_with_last_line = _run_llm(prompt, num_tokens, stop_text)
    output_with_last_line = translate_to_catalan(output_with_last_line)
    return output_with_last_line.replace(last_line, "")


# %%
def compute_metrics(sample):
    try:
        prediction = run_inference(sample['prompt'])
    except Exception as e:
        print(e)
        raise

    score = f1_score(prediction, sample['answer'])
    return {"f1": score, "predictions": prediction}


# %%
# %% [markdown]
# Eval xquad

def replace_none(row):
    for key in row:
        if row[key] is None:
            row[key] = "None"
    return row


# %%
xquad_ca = load_dataset("data", data_files="xquad_ca.csv", split="train").map(replace_none)
print("Computing xquad")
results_ca = xquad_ca.map(compute_metrics)

from pathlib import Path
Path("results").mkdir(parents=True, exist_ok=True)

results_ca.to_csv(f"results/{model_name}-xquad-ca.csv", index=False)
