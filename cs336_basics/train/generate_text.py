import argparse
import re
from dataclasses import replace
from pathlib import Path

import torch

from cs336_basics.tokenization.tokenizer import Tokenizer
from cs336_basics.train.config import default_config
from cs336_basics.train.trainer import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a training checkpoint (.pt).",
    )
    p.add_argument(
        "--vocab-path",
        type=str,
        default="cs336_basics/bpe_model_tiny_stories/bpe-vocab.txt",
        help="Path to tokenizer vocab file.",
    )
    p.add_argument(
        "--merges-path",
        type=str,
        default="cs336_basics/bpe_model_tiny_stories/bpe-merges.txt",
        help="Path to tokenizer merges file.",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default="<|endoftext|>",
        help="Prompt text. Defaults to end-of-text token.",
    )
    p.add_argument(
        "--eos-token",
        type=str,
        default="<|endoftext|>",
        help="Special token used to stop generation.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens.")
    p.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy decoding.")
    p.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling threshold in (0, 1].")
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run generation on (e.g. cpu, cuda). Defaults to cuda if available.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for generation.",
    )
    p.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override context length. If omitted, uses training default.",
    )
    p.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override number of attention heads. If omitted, uses training default.",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=None,
        help="Override RoPE theta. If omitted, uses training default.",
    )
    p.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Optional path to write generated text.",
    )
    return p.parse_args()


def _extract_model_state_dict(checkpoint_payload: dict) -> dict:
    if "model_state_dict" in checkpoint_payload:
        return checkpoint_payload["model_state_dict"]
    return checkpoint_payload


def infer_model_dims(model_state_dict: dict) -> tuple[int, int, int, int]:
    token_embedding = model_state_dict["token_embedding.weight"]
    vocab_size, d_model = token_embedding.shape

    layer_indices = []
    for key in model_state_dict:
        m = re.match(r"layers\.(\d+)\.", key)
        if m:
            layer_indices.append(int(m.group(1)))
    if not layer_indices:
        raise ValueError("Unable to infer num_layers from checkpoint state dict.")
    num_layers = max(layer_indices) + 1

    ffn_w1 = model_state_dict.get("layers.0.ffn.w1")
    if ffn_w1 is None:
        raise ValueError("Unable to infer feed-forward dimension from checkpoint state dict.")
    dim_feedforward = int(ffn_w1.shape[0])

    return int(vocab_size), int(d_model), int(num_layers), dim_feedforward


def build_cfg_from_checkpoint(args: argparse.Namespace, model_state_dict: dict):
    vocab_size, d_model, num_layers, dim_feedforward = infer_model_dims(model_state_dict)

    model_cfg = replace(
        default_config.model,
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        n_head=args.num_heads if args.num_heads is not None else default_config.model.n_head,
        theta=args.theta if args.theta is not None else default_config.model.theta,
    )

    data_cfg = replace(
        default_config.data,
        context_length=args.context_length if args.context_length is not None else default_config.data.context_length,
    )

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = replace(
        default_config.trainer,
        device=device,
        dtype=args.dtype,
    )

    return replace(default_config, model=model_cfg, data=data_cfg, trainer=trainer_cfg)


def main() -> None:
    args = parse_args()
    if not (0 < args.top_p <= 1.0):
        raise ValueError("--top-p must be in (0, 1].")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = _extract_model_state_dict(checkpoint)
    cfg = build_cfg_from_checkpoint(args, model_state_dict)

    tokenizer = Tokenizer.from_files(
        args.vocab_path,
        args.merges_path,
        special_tokens=[args.eos_token],
    )
    eos_token_id = tokenizer.special_token_to_id[args.eos_token]

    trainer = Trainer(cfg=cfg, wandb=None)
    trainer.model.load_state_dict(model_state_dict)
    trainer.model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = [eos_token_id]

    if len(prompt_ids) > cfg.data.context_length:
        prompt_ids = prompt_ids[-cfg.data.context_length :]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=cfg.trainer.device)
    with torch.inference_mode():
        output_ids = trainer.generate(
            input_ids=input_ids,
            eos_token_id=eos_token_id,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )[0].tolist()

    generated_ids = output_ids[len(prompt_ids) :]
    decoded = tokenizer.decode(generated_ids)

    print(f"prompt_tokens={len(prompt_ids)} generated_tokens={len(generated_ids)}")
    print(decoded)

    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(decoded, encoding="utf-8")
        print(f"\nSaved generation to: {output_path}")


if __name__ == "__main__":
    main()
