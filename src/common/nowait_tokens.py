"""
Pre-computed nowait token IDs for each model.

To generate token IDs for a new model, run:

python -m src.common.nowait_tokens \
    --model_name "Qwen/Qwen3-Coder-480B-A35B-Instruct" \
    --model_alias "qwen3-coder-480b-a35b-instruct"

The model_alias is used as the key in NOWAIT_TOKEN_IDS and should match
the model_id you use when running experiments (e.g., --model_id "kimi-k2-thinking").
"""

from typing import Dict, List, Optional

# Base wait tokens to suppress
WAIT_TOKENS = [
    "wait",
    "alternatively",
    "hmm",
    "but",
    "however",
    "alternative",
    "another",
    "check",
    "double-check",
    "oh",
    "maybe",
    "verify",
    "other",
    "again",
    "now",
    "ah",
    "any",
]

# Pre-computed token IDs for each model
# Key: model alias (exact match with model_id)
# Value: list of token IDs to suppress
NOWAIT_TOKEN_IDS: Dict[str, List[int]] = {
    # DeepSeek R1
    "deepseek-r1-0528": [
        949,
        1165,
        1631,
        1672,
        2020,
        4146,
        5887,
        8197,
        9547,
        11759,
        12412,
        12465,
        13079,
        13718,
        15205,
        15951,
        26284,
        30009,
        32064,
        39059,
        40544,
        41289,
        55194,
        58264,
        59065,
        72335,
        75558,
        85214,
        89961,
        92491,
        95808,
        96978,
        104253,
        115699,
        119400,
        120541,
    ],
    "kimi-k2-thinking": [
        1887,
        2165,
        2348,
        3348,
        5066,
        6440,
        7137,
        7960,
        10048,
        11567,
        13948,
        15731,
        16076,
        17651,
        17761,
        18084,
        19134,
        20343,
        30500,
        32015,
        36908,
        41021,
        43126,
        43604,
        44461,
        46839,
        48625,
        48844,
        57025,
        59031,
        63817,
        94052,
        97095,
        97600,
        106886,
        108956,
        147589,
        153124,
    ],
    "qwen3-235b-a22b-thinking-2507": [
        1466,
        1575,
        2028,
        2267,
        3328,
        3767,
        3973,
        3983,
        7039,
        8088,
        8610,
        11209,
        11409,
        11489,
        11908,
        12446,
        14037,
        14190,
        17293,
        21390,
        24765,
        29923,
        30267,
        30385,
        32627,
        32771,
        34516,
        36760,
        41963,
        43929,
        46223,
        46761,
        71032,
        75763,
        76995,
        80022,
        85395,
        92014,
        97836,
    ],
    "glm-4.6": [
        1466,
        1575,
        2028,
        2267,
        3328,
        3767,
        3973,
        3983,
        7039,
        8087,
        8609,
        11204,
        11404,
        11484,
        11902,
        12440,
        14028,
        14181,
        17272,
        21340,
        24689,
        29807,
        30150,
        30268,
        32478,
        32618,
        34346,
        36569,
        41726,
        43677,
        45944,
        46473,
        70496,
        75161,
        76383,
        79380,
        84690,
        91243,
        97009,
    ],
    "glm-4.7": [
        1466,
        1575,
        2028,
        2267,
        3328,
        3767,
        3973,
        3983,
        7039,
        8087,
        8609,
        11204,
        11404,
        11484,
        11902,
        12440,
        14028,
        14181,
        17272,
        21340,
        24689,
        29807,
        30150,
        30268,
        32478,
        32618,
        34346,
        36569,
        41726,
        43677,
        45944,
        46473,
        70496,
        75161,
        76383,
        79380,
        84690,
        91243,
        97009,
    ],
    "deepseek-v3.2": [
        949,
        1165,
        1631,
        1672,
        2020,
        4146,
        5887,
        8197,
        9547,
        11759,
        12412,
        12465,
        13079,
        13718,
        15205,
        15951,
        26284,
        30009,
        32064,
        39059,
        40544,
        41289,
        55194,
        58264,
        59065,
        72335,
        75558,
        85214,
        89961,
        92491,
        95808,
        96978,
        104253,
        115699,
        119400,
        120541,
    ],
    "minimax-m2": [
        724,
        1228,
        1683,
        2245,
        3895,
        4137,
        5617,
        7566,
        8665,
        10309,
        11269,
        11338,
        12088,
        14082,
        14998,
        21205,
        23247,
        25996,
        27972,
        29286,
        34366,
        37602,
        39976,
        43586,
        47346,
        49265,
        58909,
        61220,
        62657,
        81618,
        82145,
        86261,
        87149,
        100813,
        130526,
        134610,
        158753,
        161143,
    ],
    "qwen3-coder-480b-a35b-instruct": [
        1466,
        1575,
        2028,
        2267,
        3328,
        3767,
        3973,
        3983,
        7039,
        8088,
        8610,
        11209,
        11409,
        11489,
        11908,
        12446,
        14037,
        14190,
        17293,
        21390,
        24765,
        29923,
        30267,
        30385,
        32627,
        32771,
        34516,
        36760,
        41963,
        43929,
        46223,
        46761,
        71032,
        75763,
        76995,
        80022,
        85395,
        92014,
        97836,
    ],
    "mimo_v2.5": [
        3328,
        32771,
        11908,
        3973,
        17293,
        21390,
        3983,
        46223,
        11409,
        85395,
        80022,
        36760,
        8088,
        43929,
        12446,
        8610,
        1575,
        46761,
        97836,
        30385,
        3767,
        1466,
        30267,
        24765,
        76995,
        11209,
        34516,
        14037,
        2267,
        11489,
        29923,
        41963,
        2028,
        14190,
        92014,
        75763,
        32627,
        71032,
        7039,
    ],
    "glm-5.1": [
        3328,
        3973,
        46473,
        11404,
        3983,
        79380,
        8087,
        12440,
        75161,
        43677,
        8609,
        1575,
        34346,
        3767,
        1466,
        30268,
        11204,
        30150,
        14028,
        84690,
        36569,
        2267,
        11484,
        21340,
        32478,
        76383,
        70496,
        14181,
        32618,
        91243,
        2028,
        11902,
        29807,
        97009,
        24689,
        17272,
        45944,
        41726,
        7039,
    ],
}



def get_nowait_logit_bias(model_id: str) -> Optional[Dict[str, int]]:
    """
    Get pre-computed logit_bias dict for a model.

    Args:
        model_id: Model identifier (exact match with keys in NOWAIT_TOKEN_IDS)

    Returns:
        Dict mapping token_id (as string) to bias value (-100), or None if not found
    """
    # Exact match (case-insensitive)
    model_id_lower = model_id.lower()

    if model_id_lower in NOWAIT_TOKEN_IDS:
        token_ids = NOWAIT_TOKEN_IDS[model_id_lower]
        return {str(tid): -100 for tid in token_ids}

    print(f"⚠️  No pre-computed nowait tokens for model: {model_id}")
    print(f"   Available models: {list(NOWAIT_TOKEN_IDS.keys())}")
    print("   Run: python -m src.common.nowait_tokens --model_name <tokenizer> --model_alias <alias>")
    return None


def generate_token_ids(tokenizer_name: str) -> List[int]:
    """
    Generate nowait token IDs for a tokenizer.

    Args:
        tokenizer_name: HuggingFace tokenizer name

    Returns:
        List of token IDs
    """
    from transformers import AutoTokenizer

    # Generate all variations of wait tokens
    tokens = WAIT_TOKENS.copy()
    tokens += [t.capitalize() for t in WAIT_TOKENS]
    tokens += [t.upper() for t in WAIT_TOKENS]
    tokens += [t + p for t in WAIT_TOKENS for p in [".", ",", "!", "?", ";"]]
    tokens += [" " + t for t in tokens] + [t + " " for t in tokens]

    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_ids = []
    for token in tokens:
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid is not None and tid != tokenizer.unk_token_id:
            token_ids.append(tid)

    # Remove duplicates and sort
    token_ids = sorted(set(token_ids))
    return token_ids


def main(model_name: str, model_alias: str):
    """
    Generate and print token IDs for a model.

    Args:
        model_name: HuggingFace tokenizer name (e.g., "deepseek-ai/DeepSeek-R1-0528")
        model_alias: Alias to use as dictionary key (e.g., "deepseek-r1-0528")
    """
    token_ids = generate_token_ids(model_name)

    print(f"\n✅ Generated {len(token_ids)} token IDs")
    print(f"   Tokenizer: {model_name}")
    print(f"   Alias: {model_alias}")
    print("\nAdd this to NOWAIT_TOKEN_IDS in nowait_tokens.py:")
    print("-" * 60)
    print(f'    "{model_alias.lower()}": {token_ids},')
    print("-" * 60)


if __name__ == "__main__":
    import fire

    fire.Fire(main)