"""Одна mini-Llama. Здесь нет загрузки чужих весов."""
from transformers import LlamaConfig, LlamaForCausalLM
from common import config

def build_model(tokenizer):
    settings = config()["model"]
    model_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=settings["hidden_size"],
        intermediate_size=settings["intermediate_size"],
        num_hidden_layers=settings["num_hidden_layers"],
        num_attention_heads=settings["num_attention_heads"],
        num_key_value_heads=settings["num_attention_heads"],
        max_position_embeddings=settings["max_position_embeddings"],
        tie_word_embeddings=settings["tie_word_embeddings"],
        hidden_act="silu", rms_norm_eps=1e-5, rope_theta=10000.0,
        attention_bias=False, mlp_bias=False, attention_dropout=0.0,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False)
    model_config._attn_implementation = "sdpa"
    return LlamaForCausalLM(model_config)
