"""
Test generation from a trained checkpoint.

Usage:
    python scripts/test_generate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.data.tokenizer import ChordTokenizer
from src.model.transformer import ChordTransformer


def main(checkpoint_path: str):
    tokenizer = ChordTokenizer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = ChordTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=4,
        n_layers=4,
        max_seq_len=512,
        dropout=0.1,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}, val_loss={checkpoint['val_loss']:.4f}")
    print(f"Device: {device}")
    print("=" * 60)

    # Test prompts
    prompts = [
        "<verse> C",
        "<verse> G",
        "<chorus> F",
        "<intro> A",
        "<verse> D",
        "<verse> C F G C",
        "<verse> Amin F C G",
        "<chorus> G D Emin C",
        "<verse> D Bmin G A",
        "<intro> C Amin F G <verse> C",
    ]

    temperatures = [0.7, 1.0, 1.3]

    for prompt_str in prompts:
        prompt = tokenizer.encode(prompt_str)
        prompt = prompt[:-1]  # remove [EOS] so model continues
        prompt_ids = torch.tensor([prompt], device=device)
        decoded_prompt = tokenizer.decode(prompt)

        print(f"\nPrompt: {decoded_prompt}")
        print("-" * 40)

        for temp in temperatures:
            output = model.generate(
                prompt_ids,
                max_new_tokens=60,
                temperature=temp,
                top_k=50,
                repetition_penalty=1.2,
            )
            chords = tokenizer.decode(output[0].tolist())
            print(f"  temp={temp}: {' '.join(chords)}")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/new_model.pt")
    args = parser.parse_args()
    main(args.checkpoint)
