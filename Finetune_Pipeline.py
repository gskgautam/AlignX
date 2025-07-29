#Prompt Injection

import os
import pandas as pd

# 1) Update these as needed
ALPACA_JSON_TRAIN = ""
ALPACA_JSON_TEST  = ""
PROMPT_CSV_TRAIN  = ""
PROMPT_CSV_TEST   = ""

# 2) Sanity‑check file existence
for name, path in [
    ("ALPACA_JSON_TRAIN", ALPACA_JSON_TRAIN),
    ("ALPACA_JSON_TEST",  ALPACA_JSON_TEST),
    ("PROMPT_CSV_TRAIN",  PROMPT_CSV_TRAIN),
    ("PROMPT_CSV_TEST",   PROMPT_CSV_TEST),
]:
    print(f"{name}: {path} → exists? {os.path.exists(path)}")

# 3) Load and preview
print("\n=== Alpaca JSON Train (first 5 rows) ===")
df_alpaca_train = pd.read_json(ALPACA_JSON_TRAIN)
print(df_alpaca_train.head(), "\n")

print("=== Alpaca JSON Test (first 5 rows) ===")
df_alpaca_test = pd.read_json(ALPACA_JSON_TEST)
print(df_alpaca_test.head(), "\n")

print("=== Injection CSV Train (first 5 rows) ===")
df_prompt_train = pd.read_csv(PROMPT_CSV_TRAIN)
print(df_prompt_train.head(), "\n")

print("=== Injection CSV Test (first 5 rows) ===")
df_prompt_test = pd.read_csv(PROMPT_CSV_TEST)
print(df_prompt_test.head())

#Pipeline

import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ─── I/O paths ────────────────────────────────────────────────────────
ALPACA_JSON_TRAIN = ""
ALPACA_JSON_TEST  = ""

PROMPT_CSV_TRAIN  = ""
PROMPT_CSV_TEST   = ""


# ─── Helpers ─────────────────────────────────────────────────────────
def load_alpaca_json():
    """Load Alpaca JSON train/test."""
    df_train = pd.read_json(ALPACA_JSON_TRAIN)
    df_test  = pd.read_json(ALPACA_JSON_TEST)
    return df_train, df_test

def load_injection_csv():
    """
    Load your CSV injection prompts.
    We expect columns: instruction,input,output
    """
    df_tr = pd.read_csv(PROMPT_CSV_TRAIN)
    df_te = pd.read_csv(PROMPT_CSV_TEST)
    return df_tr, df_te

def tokenize_df(tokenizer, df: pd.DataFrame, device, max_length=512):
    """
    Tokenize and wrap into Dataset.
    - Always use df['instruction']
    - Safely get df['input'] (or empty string if missing)
    """
    # 1) grab instructions
    instrs = df["instruction"].astype(str).tolist()
    # 2) safe inputs: if no 'input' column, use empty strings
    if "input" in df.columns:
        inputs = df["input"].fillna("").astype(str).tolist()
    else:
        inputs = [""] * len(df)

    # 3) tokenize
    enc = tokenizer(
        instrs,
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    # 4) for causal LM, labels=input_ids
    enc["labels"] = enc["input_ids"].clone()

    # 5) move to device
    for k, v in enc.items():
        enc[k] = v.to(device)

    return Dataset.from_dict(enc)


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f" Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")

def train_adapter(
    base_model, tokenizer, train_ds, eval_ds, adapter_name, device,
    num_epochs=3, batch_size=2, lr=5e-4
):
    """
    Attach a new LoRA adapter onto `base_model`, train it on train_ds,
    eval on eval_ds, and save under ./results/{adapter_name}.
    Returns the fused model (base + this new adapter).
    """
    # 1) configure LoRA
    lora_cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(base_model, lora_cfg).to(device)
    print(f"\n Training LoRA adapter `{adapter_name}`")
    print_trainable_parameters(model)

    # 2) Trainer
    args = TrainingArguments(
        output_dir=f"./results/{adapter_name}",
        save_dir  =f"./results/{adapter_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_dir=f"./logs/{adapter_name}",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_steps=500,
        save_total_limit=2,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    # 3) Train
    trainer.train()

    # 4) sanity‐check some norms
    print("Checking updated LoRA layers:")
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(f" • {n:60} ∥ {torch.norm(p).item(): .4f}")

    # 5) persist
    model.save_pretrained(f"./results/{adapter_name}")
    return model

def compute_task_vector(base_model, adapted_model, out_path="task_vector.pt"):
    """
    Compute the difference (Adapter – Base) for every shared weight,
    save as a pt tensor dict.
    """
    diff = {}
    base_sd = base_model.state_dict()
    adapt_sd = adapted_model.state_dict()
    for k in adapt_sd:
        if k in base_sd:
            d = adapt_sd[k] - base_sd[k]
            if torch.norm(d) > 0:
                diff[k] = d
    if not diff:
        raise RuntimeError("No differences found! Did adapters actually train?")
    torch.save(diff, out_path)
    print(f"Task vector saved → `{out_path}` ({len(diff)} layers)")
    return diff

def apply_task_vector(base_model, task_vector, gamma=1.0):
    """
    Add γ·Δ to every base‐weight in the model and reload.
    """
    sd = base_model.state_dict()
    for k,v in task_vector.items():
        sd[k] = sd[k] + gamma * v
    base_model.load_state_dict(sd)
    return base_model

# ─── Main ────────────────────────────────────────────────────────────
def main():
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # tokenizer & base_model
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)

    # ── Stage 1: Alpaca JSON ───────────────────────────────────────────
    df_train_j, df_test_j = load_alpaca_json()
    ds_train_j = tokenize_df(tok, df_train_j, device)
    ds_eval_j  = tokenize_df(tok, df_test_j,  device)

    adapter1 = train_adapter(
        base_model=base,
        tokenizer=tok,
        train_ds=ds_train_j,
        eval_ds=ds_eval_j,
        adapter_name="alpaca_helpfulness",
        device=device,
    )

# # Step:1


import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import Dataset

# ─── I/O paths ────────────────────────────────────────────────────────
ALPACA_JSON_TRAIN = ""
ALPACA_JSON_TEST  = ""
PROMPT_CSV_TRAIN  = ""
PROMPT_CSV_TEST   = ""

# ─── Helpers ─────────────────────────────────────────────────────────
def load_alpaca_json():
    print("[1/8] Loading Alpaca JSON files…")
    df_train = pd.read_json(ALPACA_JSON_TRAIN)
    df_test  = pd.read_json(ALPACA_JSON_TEST)
    print(f"    • Train size: {len(df_train)}, Test size: {len(df_test)}")
    return df_train, df_test

def load_injection_csv():
    print("[2/8] Loading injection CSV files…")
    df_tr = pd.read_csv(PROMPT_CSV_TRAIN)
    df_te = pd.read_csv(PROMPT_CSV_TEST)
    print(f"    • Inject‑Train rows: {len(df_tr)}, Inject‑Test rows: {len(df_te)}")
    return df_tr, df_te

def get_single_prompt_col(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()
    if len(cols) != 1:
        raise ValueError(f"Expected exactly one column in injection CSV, got: {cols}")
    return cols[0]

def prepend_col(df: pd.DataFrame, prompt_col: str) -> pd.DataFrame:
    print("[3/8] Prepending Chain‑of‑Thought prefix to column", prompt_col)
    df = df.copy()
    df[prompt_col] = (
        "Chain of Thought: Let's think step by step. "
        + df[prompt_col].astype(str)
    )
    return df

def tokenize_df(tokenizer, df: pd.DataFrame, prompt_col: str, device, max_length=512):
    print(f"[4/8] Tokenizing {len(df)} examples (prompt='{prompt_col}')…")
    instrs = df[prompt_col].astype(str).tolist()
    if "input" in df.columns:
        inputs = df["input"].fillna("").astype(str).tolist()
    else:
        inputs = [""] * len(df)
    enc = tokenizer(
        instrs,
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc:
        enc[k] = enc[k].to(device)
    print("    • Tokenization complete.")
    return Dataset.from_dict(enc)

# ─── New wrapper for JSON-based Alpaca splits ─────────────────────────
def tokenize_json_df(tokenizer, df: pd.DataFrame, device, max_length=512):
    return tokenize_df(tokenizer, df, "instruction", device, max_length)

def print_trainable_parameters(model):
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f" Trainable params: {trainable}/{total} ({100*trainable/total:.2f}%)")

def train_adapter(
    base_model, tokenizer, train_ds, eval_ds, adapter_name, device,
    num_epochs=3, batch_size=2, lr=5e-4
):
    print(f"\n Training LoRA adapter `{adapter_name}`")
    lora_cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(base_model, lora_cfg).to(device)
    print_trainable_parameters(model)

    args = TrainingArguments(
        output_dir=f"./results/{adapter_name}",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_dir=f"./logs/{adapter_name}",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        do_eval=True
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    trainer.train()
    print(f"    Finished training `{adapter_name}`.")

    print("    Checking updated LoRA layers:")
    for n,p in model.named_parameters():
        if p.requires_grad:
            print(f"      • {n:60} ∥ norm {torch.norm(p).item(): .4f}")

    model.save_pretrained(f"./results/{adapter_name}")
    print(f"   Adapter saved to ./results/{adapter_name}")
    return model

def compute_task_vector(base_model, adapted_model, out_path="task_vector.pt"):
    print("[6/8] Computing task vector (difference between base and adapter)…")
    diff = {}
    base_sd = base_model.state_dict()
    adapt_sd = adapted_model.state_dict()
    for k in adapt_sd:
        if k in base_sd:
            d = adapt_sd[k] - base_sd[k]
            if torch.norm(d) > 0:
                diff[k] = d
    if not diff:
        raise RuntimeError("No differences found! Did adapters actually train?")
    torch.save(diff, out_path)
    print(f"    Task vector saved → `{out_path}` ({len(diff)} layers)")
    return diff

def apply_task_vector(base_model, task_vector, gamma=1.0):
    print("[7/8] Applying task vector to base model…")
    sd = base_model.state_dict()
    for k,v in task_vector.items():
        sd[k] = sd[k] + gamma * v
    base_model.load_state_dict(sd)
    print("    • Task vector applied.")
    return base_model

# ─── Main ────────────────────────────────────────────────────────────
def main():
    print("[0/8] Starting training pipeline…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    print("[1/8] Loading tokenizer and base model…")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    print("    Base model loaded.")

    # Stage 1
    df_train_j, df_test_j = load_alpaca_json()
    ds_train_j = tokenize_json_df(tok, df_train_j, device)
    ds_eval_j  = tokenize_json_df(tok, df_test_j,  device)
    adapter1   = train_adapter(base, tok, ds_train_j, ds_eval_j, "alpaca_helpfulness", device)

    # Stage 2
    df_tr_csv, df_te_csv = load_injection_csv()
    prompt_col = get_single_prompt_col(df_tr_csv)

    df_tr_csv = prepend_col(df_tr_csv, prompt_col)
    df_te_csv = prepend_col(df_te_csv, prompt_col)

    ds_train_col = tokenize_df(tok, df_tr_csv, prompt_col, device)
    ds_eval_col  = tokenize_df(tok, df_te_csv,  prompt_col, device)
    adapter2     = train_adapter(adapter1, tok, ds_train_col, ds_eval_col, "alpaca_inject_col", device)

    # Task Vector & final save
    tv    = compute_task_vector(base, adapter2)
    final = apply_task_vector(base, tv, gamma=1.0)
    final.save_pretrained("./results/final_model")
    print("[8/8] 🎉 Final model saved under `./results/final_model`")

if __name__=="__main__":
    main()


# >>> Continue from after “Adapter saved to ./results/alpaca_helpfulness”

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, PeftModel, LoraConfig
from datasets import Dataset

# 0) Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ADAPTER1_PATH    = ""
PROMPT_CSV_TRAIN = ""
PROMPT_CSV_TEST  = ""

# 1) Reload tokenizer and adapter1
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
).to(device)
adapter1 = PeftModel.from_pretrained(base_model, ADAPTER1_PATH).to(device)

# 2) Prepare Stage 2 datasets again
df_tr = pd.read_csv(PROMPT_CSV_TRAIN)
df_te = pd.read_csv(PROMPT_CSV_TEST)
prompt_col = df_tr.columns[0]

# Prepend 
df_tr[prompt_col] = "Chain of Thought: Let's think step by step. " + df_tr[prompt_col].astype(str)
df_te[prompt_col] = "Chain of Thought: Let's think step by step. " + df_te[prompt_col].astype(str)

# Tokenize helper
def make_dataset(df):
    enc = tokenizer(
        df[prompt_col].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc:
        enc[k] = enc[k].to(device)
    return Dataset.from_dict(enc)

ds_tr_col = make_dataset(df_tr)
ds_te_col = make_dataset(df_te)

# 3) Train second adapter with remove_unused_columns=False
lora_cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
model2 = get_peft_model(adapter1, lora_cfg).to(device)

args2 = TrainingArguments(
    output_dir="./results/alpaca_inject",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    do_eval=True,
    remove_unused_columns=False   # <— prevents the Trainer error
)

trainer2 = Trainer(
    model=model2,
    args=args2,
    train_dataset=ds_tr_col,
    eval_dataset=ds_te_col,
)

trainer2.train()
model2.save_pretrained("./results/alpaca_inject")
print("Adapter saved to ./results/alpaca_inject")



import torch
import gc
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ─── Setup ────────────────────────────────────────────────────────────
print("Cleaning up and setting device…")
gc.collect()
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ─── 1) Load base model ──────────────────────────────────────────────
print("1/5 Loading base Llama-2-7b model…")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf"
).to(device)
print("  Base model loaded.\n")

# ─── 2) Load the second adapter ───────────────────────────────────────
print("2/5 Loading PEFT adapter 'alpaca_inject'…")
adapter2 = PeftModel.from_pretrained(
    base_model,
    "./results/alpaca_inject"
).to(device)
print("   Adapter loaded.\n")

# ─── 3) Merge adapter into base ───────────────────────────────────────
print("3/5 Merging adapter into base weights…")
merged_model = adapter2.merge_and_unload()
merged_model.to(device)
print("   Adapter merged.\n")

# ─── 4) Compute and save task vector ──────────────────────────────────
print("4/5 Computing task vector (merged – base)…")
base_sd   = base_model.state_dict()
merged_sd = merged_model.state_dict()
task_vector = {}
for k, v in merged_sd.items():
    if k in base_sd:
        delta = v - base_sd[k]
        if torch.norm(delta) > 0:
            task_vector[k] = delta.cpu()
torch.save(task_vector, "task_vector.pt")
print(f"   Task vector saved ( {len(task_vector)} layers )\n")

# ─── 5) Apply task vector and save final model ────────────────────────
print("5/5 Applying task vector and saving final fused model…")
sd = base_model.state_dict()
for k, v in task_vector.items():
    sd[k] = sd[k] + v.to(device)
base_model.load_state_dict(sd)
base_model.save_pretrained("./results/final_model")
print("  Final fused model saved under './results/final_model'")


# # Step 2A: Train a new LoRA adapter on the BeaverTails prompts



import torch, gc, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import Dataset

# ── 0) Cleanup & device ───────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}\n")

# ── 1) Reload the fused model from Step 1 ─────────────────────────────
print("1/5 Loading fused model from Step 1…")
base_model = AutoModelForCausalLM.from_pretrained("./results/final_model").to(device)
print(" Base model loaded.\n")

# ── 1.1) Load tokenizer ──────────────────────────────────────────────
print("1.1 Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
print("    Tokenizer ready.\n")

# ── 2) Load BeaverTails CSVs ──────────────────────────────────────────
print("2/5 Reading BeaverTails train/test CSVs…")
bt_train = pd.read_csv("")
bt_test  = pd.read_csv("")
print(f"   • Train rows: {len(bt_train)}, Test rows: {len(bt_test)}\n")

# ── 3) Detect prompt column & prepend ────────────────────────────
prompt_col = bt_train.columns[0]
print(f"3/5 Using single prompt column: '{prompt_col}'")
print("   prefix added.\n")

# ── 4) Tokenize into Dataset ───────────────────────────────────────
def make_dataset(df):
    enc = tokenizer(
        df[prompt_col].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc: enc[k] = enc[k].to(device)
    return Dataset.from_dict(enc)

print("4/5 Tokenizing…")
ds_bt_tr = make_dataset(bt_train)
ds_bt_te = make_dataset(bt_test)
print("   Tokenization complete.\n")

# ── 5) Train second adapter (BeaverTails) ─────────────────────────────
print("5/5 Training BeaverTails LoRA adapter…")
lora_cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
adapter_bt = get_peft_model(base_model, lora_cfg).to(device)

args_bt = TrainingArguments(
    output_dir="./results/beavertails_adapter1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    do_eval=True,
    remove_unused_columns=False
)
trainer_bt = Trainer(
    model=adapter_bt,
    args=args_bt,
    train_dataset=ds_bt_tr,
    eval_dataset=ds_bt_te,
)
trainer_bt.train()
adapter_bt.save_pretrained("./results/beavertails_adapter1")
print("   BeaverTails adapter saved to ./results/beavertails_adapter1")



# >>> Continue from after “BeaverTails adapter saved to ./results/beavertails_adapter1”

import torch
import gc
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, PeftModel, LoraConfig
from datasets import Dataset

# ── 0) Cleanup & device ───────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ── 1) Reload base + BeaverTails adapter from Block 1 ─────────────────
print("1/6 Loading base model and BeaverTails adapter from Block 1…")
tokenizer  = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("./results/final_model").to(device)
adapter_bt1 = PeftModel.from_pretrained(base_model, "./results/beavertails_adapter1").to(device)
print("Loaded base + adapter1.\n")

# ── 2) Load BeaverTails CSVs again ────────────────────────────────────
print("2/6 Reading BeaverTails train/test CSVs…")
df_tr = pd.read_csv("")
df_te = pd.read_csv("")
print(f"   • Train rows: {len(df_tr)}, Test rows: {len(df_te)}\n")

# ── 3) Prepend ───────────────────────────────────────────────────
prompt_col = df_tr.columns[0]
print(f"3/6 Prepending to column '{prompt_col}'")
print("   prefix added.\n")

# ── 4) Tokenize ──────────────────────────────────────────────────────
def make_dataset(df):
    enc = tokenizer(
        df[prompt_col].tolist(),
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc: enc[k] = enc[k].to(device)
    return Dataset.from_dict(enc)

print("4/6 Tokenizing…")
ds_tr_bt = make_dataset(df_tr)
ds_te_bt = make_dataset(df_te)
print("  Tokenization complete.\n")

# ── 5) Train second BeaverTails adapter ───────────────────────────────
print("5/6 Training second BeaverTails LoRA adapter…")
lora_cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
adapter_bt2 = get_peft_model(adapter_bt1, lora_cfg).to(device)

args2 = TrainingArguments(
    output_dir="./results/beavertails_adapter2",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    do_eval=True,
    remove_unused_columns=False,
)
trainer2 = Trainer(
    model=adapter_bt2,
    args=args2,
    train_dataset=ds_tr_bt,
    eval_dataset=ds_te_bt,
)
trainer2.train()
adapter_bt2.save_pretrained("./results/beavertails_adapter2")
print("   BeaverTails adapter2 saved to ./results/beavertails_adapter2\n")

# ── 6) Summary ────────────────────────────────────────────────────────
print("Block 2 complete: new adapter at ./results/beavertails_adapter2")


import torch, gc
from transformers import AutoModelForCausalLM
from peft import PeftModel

# ── 0) Cleanup & device ───────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}\n")

# ── 1) Load the fused model from Step 1 and then the 2nd BeaverTails adapter ───
print("1/4 Loading base fused model and BeaverTails adapter2…")
base = AutoModelForCausalLM.from_pretrained("./results/final_model").to(device)
peft_bt2 = PeftModel.from_pretrained(base, "./results/beavertails_adapter2").to(device)
print("   Loaded base + adapter2.\n")

# ── 2) Merge & unload the PEFT adapter ────────────────────────────────────
print("2/4 Merging BeaverTails adapter2 into base…")
merged = peft_bt2.merge_and_unload().to(device)
del peft_bt2; gc.collect(); torch.cuda.empty_cache()
print("   Merge complete.\n")

# ── 3) Compute task vector ───────────────────────────────────────────────
print("3/4 Computing BeaverTails task vector…")
bv = {}
base_sd   = base.state_dict()
merged_sd = merged.state_dict()
for k, v in merged_sd.items():
    if k in base_sd:
        delta = v - base_sd[k]
        if torch.norm(delta) > 0:
            bv[k] = delta.cpu()
torch.save(bv, "./results/beavertails_task_vector.pt")
print(f"   Saved task vector ({len(bv)} layers)\n")

# ── 4) Apply task vector & save final fused model ───────────────────────
print("4/4 Applying task vector and saving final model…")
sd = base.state_dict()
for k, v in bv.items():
    sd[k] = sd[k] + v.to(device)
base.load_state_dict(sd)
out_dir = "./results/final_model_beavertails"
base.save_pretrained(out_dir)
print(f"   🎉 Final model with BeaverTails saved to {out_dir}")



# >>> Step 3 – Block 1: Train first TruthfulQA (Honesty) adapter

import torch, gc, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import Dataset

# 0) Cleanup & device
print(" Clearing GPU cache…")
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}\n")

# 1) Reload both fused models from Step 1 & Step 2
print("1/6 Loading Alpaca‐only and Alpaca+BeaverTails fused models…")
alpaca_model = AutoModelForCausalLM.from_pretrained("./results/final_model").to(device)
beaver_model = AutoModelForCausalLM.from_pretrained("./results/final_model_beavertails").to(device)
print("   Both base models loaded.\n")

# 2) Load tokenizer
print("2/6 Loading tokenizer…")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
print("   Tokenizer ready.\n")

# 3) Read TruthfulQA CSVs
print("3/6 Reading TruthfulQA train/test CSVs…")
qa_train = pd.read_csv("")
qa_test  = pd.read_csv("")
print(f"   Train rows: {len(qa_train)}, Test rows: {len(qa_test)}\n")

# 4) Prepend 
prompt_col = qa_train.columns[0]
print(f"4/6 Prepending to column '{prompt_col}'…")
print("   added.\n")

# 5) Tokenize helper
def make_ds(df, model_name):
    print(f"   • Tokenizing for {model_name}…")
    enc = tokenizer(
        df[prompt_col].tolist(),
        padding="max_length", truncation=True, max_length=512,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc: enc[k] = enc[k].to(device)
    return Dataset.from_dict(enc)

# produce datasets once
ds_tr = make_ds(qa_train, "TruthfulQA")
ds_te = make_ds(qa_test,  "TruthfulQA")
print("   Tokenization complete.\n")

# 6) Fine‐tune two adapters in parallel
for base, tag in [(alpaca_model, "alpaca"), (beaver_model, "beavertails")]:
    print(f"Training Honesty adapter on '{tag}' base…")
    cfg = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
    adapter = get_peft_model(base, cfg).to(device)

    out_dir = f"./results/step3/truthfulqa_honesty1_{tag}"
    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        do_eval=True,
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=adapter, args=args,
        train_dataset=ds_tr, eval_dataset=ds_te
    )
    trainer.train()
    adapter.save_pretrained(out_dir)
    print(f"   Saved adapter1_{tag} → {out_dir}\n")



# >>> Step 3 – Block 2: Injected prompts training

import torch, gc, pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import get_peft_model, PeftModel, LoraConfig
from datasets import Dataset

# 0) Cleanup & device
print(" Clearing GPU cache…")
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}\n")

# 1) Reload base + adapter1
print("1/6  Loading base + first Honesty adapter…")
tokenizer  = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained("./results/final_model_beavertails").to(device)
adapter_h1 = PeftModel.from_pretrained(base_model, "./results/step3/truthfulqa_honesty1").to(device)
print("   Loaded adapter1.\n")

# 2) Read injection CSVs
print("2/6 Reading injected TQA prompts…")
inj_tr = pd.read_csv("")
inj_te = pd.read_csv("")
print(f"    Rows: {len(inj_tr)} / {len(inj_te)}\n")

# 3) Prepend 
prompt_col = inj_tr.columns[0]
print(f"3/6  Prepending to '{prompt_col}'")
print("   added.\n")

# 4) Tokenize
def mk_ds(df):
    enc = tokenizer(
        df[prompt_col].tolist(),
        padding="max_length", truncation=True, max_length=512,
        return_tensors="pt"
    )
    enc["labels"] = enc["input_ids"].clone()
    for k in enc: enc[k] = enc[k].to(device)
    return Dataset.from_dict(enc)

print("4/6 Tokenizing…")
ds_i_tr = mk_ds(inj_tr)
ds_i_te = mk_ds(inj_te)
print("   Tokenization done.\n")

# 5) Train second honesty adapter
print("5/6 Training second Honesty adapter…")
cfg2 = LoraConfig(r=64, lora_alpha=32, lora_dropout=0.1)
adapter_h2 = get_peft_model(adapter_h1, cfg2).to(device)

args2 = TrainingArguments(
    output_dir="./results/step3/truthfulqa_honesty2",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=5e-4,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    do_eval=True,
    remove_unused_columns=False
)
trainer2 = Trainer(
    model=adapter_h2, args=args2,
    train_dataset=ds_i_tr, eval_dataset=ds_i_te
)
trainer2.train()
adapter_h2.save_pretrained("./results/step3/truthfulqa_honesty2")
print("   Adapter2 saved to ./results/step3/truthfulqa_honesty2\n")

# 6) Summary
print("Block 2 complete: adapters at step3 folders.")



# >>> Step 3 – Block 3: Merge adapters, compute task vector, save final

import torch, gc
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 0) Cleanup & device
gc.collect(); torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Device: {device}\n")

# 1) Load base and second adapter
print("1/4  Loading base + second Honesty adapter…")
base = AutoModelForCausalLM.from_pretrained("./results/final_model_beavertails").to(device)
peft_h2 = PeftModel.from_pretrained(base, "./results/step3/truthfulqa_honesty2").to(device)
print("   Adapter2 loaded.\n")

# 2) Merge & unload
print("2/4  Merging Honesty adapter…")
merged = peft_h2.merge_and_unload().to(device)
del peft_h2; gc.collect(); torch.cuda.empty_cache()
print("    Merge done.\n")

# 3) Compute Task‑Vector
print("3/4  Computing Honesty task vector…")
tv = {}
base_sd   = base.state_dict()
merged_sd = merged.state_dict()
for k, v in merged_sd.items():
    if k in base_sd:
        delta = v - base_sd[k]
        if torch.norm(delta) > 0:
            tv[k] = delta.cpu()
torch.save(tv, "./results/step3/truthfulqa_task_vector.pt")
print(f"   Task vector saved ({len(tv)} layers)\n")

# 4) Apply & save final fused model
print("4/4 Applying task vector & saving final model…")
sd = base_sd
for k, v in tv.items():
    sd[k] = sd[k] + v.to(device)
base.load_state_dict(sd)
out = "./results/step3/final_model_step3"
base.save_pretrained(out)
print(f" Final fused model saved to {out}")

