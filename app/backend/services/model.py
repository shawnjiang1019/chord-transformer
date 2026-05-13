"""
Model and tokenizer singletons.

Both are loaded once on first use and reused across requests.
Run uvicorn from the repo root so that relative paths resolve correctly:

    uvicorn app.backend.main:app --reload --port 8000
"""

from pathlib import Path
import yaml
import torch

from src.data.tokenizer import ChordTokenizer
from src.model.transformer import ChordTransformer

_CHECKPOINT = Path("checkpoints/best_model.pt")
_CONFIG = Path("configs/model_config.yaml")

_tokenizer: ChordTokenizer | None = None
_model: ChordTransformer | None = None


def get_tokenizer() -> ChordTokenizer:
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = ChordTokenizer()
    return _tokenizer


def get_model() -> ChordTransformer:
    global _model
    if _model is None:
        tok = get_tokenizer()

        cfg: dict = {}
        if _CONFIG.exists():
            with open(_CONFIG) as f:
                cfg = yaml.safe_load(f).get("model", {})

        model = ChordTransformer(
            vocab_size=tok.vocab_size,
            d_model=cfg.get("d_model", 256),
            n_heads=cfg.get("n_heads", 4),
            n_layers=cfg.get("n_layers", 4),
            max_seq_len=cfg.get("max_seq_len", 512),
            dropout=0.0,  # inference only
        )

        if _CHECKPOINT.exists():
            ckpt = torch.load(_CHECKPOINT, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])

        model.eval()
        _model = model

    return _model


def run_inference(
    input_ids: list[int],
    num_chords: int,
    temperature: float,
    top_k: int,
) -> list[int]:
    tok = get_tokenizer()
    model = get_model()

    x = torch.tensor([input_ids])
    eos_id = tok.token2id["[EOS]"]

    output = model.generate(
        x,
        max_new_tokens=num_chords * 3,  # 3 tokens per chord
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )

    return output[0].tolist()
