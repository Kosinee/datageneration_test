import os
import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class GenericModel:
    def __init__(self, hf_model_name, vllm_params, sampling_params, max_tokens=512):
        self.hf_model_name = hf_model_name

        # Prevent duplicate trust_remote_code
        vllm_params = dict(vllm_params)  # copy to avoid mutating original
        vllm_params.pop("trust_remote_code", None)

        self.model = LLM(
            model=hf_model_name,
            dtype="auto",
            tokenizer_mode="auto",
            trust_remote_code=True,
            **vllm_params
        )
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            **sampling_params
        )

    def score(self, prompts):
        try:
            texts_formatted = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False
                )
                for p in prompts
            ]
        except Exception:
            texts_formatted = prompts

        raw_ans = self.model.generate(
            texts_formatted,
            sampling_params=self.sampling_params,
            use_tqdm=True
        )
        return [a.outputs[0].text for a in raw_ans]

    def score_file(self, prompts, output_file):
        responses = self.score(prompts)
        with open(output_file, "w", encoding="utf-8") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.json")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    models_cfg_all = cfg["models"]
    model_keys = list(models_cfg_all.keys())


    model_key = model_keys[0]
    m = models_cfg_all[model_key]

    prompts = cfg["prompts"]
    out_dir = Path(cfg["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)


    model = GenericModel(
        hf_model_name=m["hf_model_name"],
        vllm_params=m.get("vllm_params", {}),
        sampling_params=m.get("sampling_params", {}),
        max_tokens=m.get("max_tokens", 512)
    )
    out_path = out_dir / f"{model_key}.jsonl"
    model.score_file(prompts, out_path)
    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
