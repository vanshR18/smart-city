"""
dataset.py
──────────
Handles loading, tokenizing, and packaging data for DistilBERT training.

Key concepts:
- Tokenizer converts raw text → input_ids + attention_mask tensors
- EmergencyDataset wraps a DataFrame into a PyTorch Dataset
- DataCollator handles dynamic padding (pad to longest in batch, not max_length)
  This saves memory vs padding everything to 512.
"""



import pandas as pd
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast, DataCollatorWithPadding
from pathlib import Path
from loguru import logger
from typing import Optional

# ── Constants 
MODEL_CHECKPOINT = "distilbert-base-uncased"   # 66M params, 2x faster than BERT-base
MAX_LENGTH       = 128    # emergency texts are short; 128 tokens is enough
DATA_DIR         = Path(__file__).parent / "data"

LABEL2ID = {
    "ACCIDENT": 0,
    "FIRE":     1,
    "FLOOD":    2,
    "CRIME":    3,
    "CROWD":    4,
    "MEDICAL":  5,
    "NORMAL":   6,
}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS  = len(LABEL2ID)


# ── Tokenizer singleton 
# Load once and reuse — tokenizer init is slow (reads vocab from HuggingFace)
_tokenizer: Optional[DistilBertTokenizerFast] = None

def get_tokenizer() -> DistilBertTokenizerFast:
    global _tokenizer
    if _tokenizer is None:
        logger.info(f"Loading tokenizer: {MODEL_CHECKPOINT}")
        _tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_CHECKPOINT)
    return _tokenizer


# ── Dataset class 
class EmergencyDataset(Dataset):
    """
    PyTorch Dataset wrapping a pandas DataFrame of (text, label) pairs.

    Each __getitem__ returns:
      input_ids      → token ID list
      attention_mask → 1 for real token, 0 for padding
      labels         → integer class index (0–6)
    """

    def __init__(self, df: pd.DataFrame, tokenizer: DistilBertTokenizerFast):
        self.labels = df["label"].tolist()

        # Tokenize all texts upfront — much faster than tokenizing in __getitem__
        logger.info(f"Tokenizing {len(df)} samples...")
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,    # cut text longer than MAX_LENGTH
            max_length=MAX_LENGTH,
            padding=False,      # DataCollator pads per-batch (more efficient)
        )
        logger.info("Tokenization complete")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx],
        }


# ── Loaders 
def load_datasets() -> tuple["EmergencyDataset", "EmergencyDataset", "EmergencyDataset"]:
    """
    Loads train/val/test CSVs → three EmergencyDataset objects.
    Run download_dataset.py first if CSVs are missing.
    """
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing: {path}\nRun:  python scripts/download_dataset.py"
            )

    tokenizer = get_tokenizer()
    train_df  = pd.read_csv(DATA_DIR / "train.csv")
    val_df    = pd.read_csv(DATA_DIR / "val.csv")
    test_df   = pd.read_csv(DATA_DIR / "test.csv")

    logger.info(f"Splits — Train:{len(train_df)} | Val:{len(val_df)} | Test:{len(test_df)}")

    return (
        EmergencyDataset(train_df, tokenizer),
        EmergencyDataset(val_df,   tokenizer),
        EmergencyDataset(test_df,  tokenizer),
    )


def get_data_collator() -> DataCollatorWithPadding:
    """Dynamic padding — each batch padded to its own longest sequence."""
    return DataCollatorWithPadding(tokenizer=get_tokenizer(), return_tensors="pt")