"""Local ProtT5 per-residue embedding extraction via HuggingFace transformers.

Requires: torch, transformers
Model: Rostlab/prot_t5_xl_uniref50 (1024-dim per residue)
"""

import re
import sys

import numpy as np
import torch
from tqdm import tqdm

from .device import get_device


def load_prott5(
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    device: torch.device | None = None,
):
    """Load the ProtT5 tokenizer + encoder once, for reuse across many
    ``extract_prott5`` calls (pass the returned tuple as ``loaded=``). Avoids
    re-loading the ~2.5 GB model per chunk in shardable/streaming callers."""
    from transformers import AutoTokenizer, T5EncoderModel

    if device is None:
        device = get_device()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except ImportError:
        # The *fast* T5 tokenizer is built by converting the SentencePiece model,
        # which pulls in `protobuf`. Where protobuf isn't installed (some environments
        # ship sentencepiece but not protobuf), fall back to the slow SentencePiece
        # tokenizer: for ProtT5's space-separated single-residue input the two yield
        # identical token ids, so embeddings are unchanged.
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = T5EncoderModel.from_pretrained(model_name).to(device)
    model.eval()
    print(f"Loaded {model_name} (dim={model.config.d_model}) on {device}")
    return tokenizer, model, device


def extract_prott5(
    fasta_dict: dict[str, str],
    model_name: str = "Rostlab/prot_t5_xl_uniref50",
    batch_size: int = 4,
    device: torch.device | None = None,
    max_residues: int = 2000,
    loaded: tuple | None = None,
) -> dict[str, np.ndarray]:
    """Extract per-residue embeddings from ProtT5 encoder.

    ProtT5 is a T5 encoder with *relative* position bias — it has NO architectural
    sequence-length limit, so we never truncate. ``max_residues`` is only an
    out-of-memory guard: sequences above it are SKIPPED loudly, never silently
    chopped. (This mirrors ``tools/embeddings/dispatch.py::_extract_upstream``,
    which defaults to the same 2000 cap with skip-not-truncate semantics.)

    History note: this wrapper previously called the tokenizer with
    ``truncation=True, max_length=512`` — stock HuggingFace BERT-era boilerplate
    that does NOT apply to T5. It silently truncated every sequence > 511 aa to
    its first 511 residues while still slicing ``[:seq_len]`` with the full
    length, returning short, wrong embeddings with no error. Removed 2026-06-04.

    Parameters
    ----------
    fasta_dict : dict
        {protein_id: amino_acid_sequence} — must be UNGAPPED (strip alignment
        gaps with ``fasta_io.degap`` first).
    model_name : str
        HuggingFace model identifier.
    batch_size : int
        Sequences per forward pass. Keep small (2-8) for long sequences.
    device : torch.device, optional
        Defaults to CUDA if available, else MPS, else CPU.
    max_residues : int
        Out-of-memory skip guard. Sequences strictly longer than this are
        skipped (logged to stderr), not truncated. Default 2000.

    Returns
    -------
    dict
        {protein_id: np.ndarray of shape (L, 1024)} float32. Skipped sequences
        are absent from the returned dict.
    """
    if loaded is not None:
        tokenizer, model, device = loaded
    else:
        tokenizer, model, device = load_prott5(model_name, device)
    embed_dim = model.config.d_model

    # Skip-don't-truncate: drop empty + over-length sequences up front, loudly.
    skipped = [sid for sid, s in fasta_dict.items() if len(s) > max_residues]
    for sid in skipped:
        print(
            f"⚠️  Skipping {sid} (len {len(fasta_dict[sid])} > max_residues "
            f"{max_residues}) — would risk OOM; NOT truncated.",
            file=sys.stderr,
        )
    ids = [sid for sid, s in fasta_dict.items() if 0 < len(s) <= max_residues]
    # Length-sorted batching keeps each padded batch tight, so a lone long
    # sequence doesn't inflate padding (and transient attention memory) for the
    # whole batch.
    ids.sort(key=lambda sid: len(fasta_dict[sid]))

    embeddings: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(ids), batch_size), desc="ProtT5"):
        batch_ids = ids[i : i + batch_size]
        batch_seqs = []
        for sid in batch_ids:
            seq = re.sub(r"[UZOB]", "X", fasta_dict[sid])
            batch_seqs.append(" ".join(list(seq)))

        encoded = tokenizer(
            batch_seqs,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        for j, sid in enumerate(batch_ids):
            seq_len = len(fasta_dict[sid])
            # Guard the old silent-truncation footgun: the hidden state must hold
            # at least seq_len positions. If any cap ever sneaks back into the
            # tokenizer call, fail LOUDLY here rather than return a short tensor.
            assert output.last_hidden_state.shape[1] >= seq_len, (
                f"{sid}: hidden state has {output.last_hidden_state.shape[1]} "
                f"positions but sequence is {seq_len} aa — truncation regression!"
            )
            emb = output.last_hidden_state[j, :seq_len].cpu().numpy().astype(np.float32)
            embeddings[sid] = emb

    msg = f"Extracted {len(embeddings)} proteins, dim={embed_dim}"
    if skipped:
        msg += f" ({len(skipped)} skipped, len > {max_residues} aa)"
    print(msg)
    return embeddings
