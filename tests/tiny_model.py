"""Builds a tiny Qwen-style model + tokenizer offline, so tests run without downloads."""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


def make_tiny_model(out: str | Path, corpus: list[str] | None = None, hidden: int = 64, layers: int = 4) -> Path:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    corpus = corpus or ["hello world this is a tiny test corpus for a tiny model"] * 10
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    specials = ["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=800, special_tokens=specials,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>", additional_special_tokens=["<|im_start|>"])
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)

    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(out)
    return out
