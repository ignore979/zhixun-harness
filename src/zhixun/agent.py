"""A real PyTRIO policy behind a finite action catalog and deterministic fallback."""
from __future__ import annotations

import os

from zhixun.data import ROOT
from zhixun.fair_core import build_prompt, rule_action, validate_action
from zhixun.llm_policy import parse_decision
from zhixun.pytrio_training import prompt_tokens


class LlmPolicy:
    def __init__(self, config, *, model, model_path=None, calls=0, before_request=None,
                 client=None, tokenizer=None):
        self.config, self.calls = config, calls
        self.before_request = before_request
        self.client, self.tokenizer, self.initialization_error = client, tokenizer, None
        if client is None:
            try:
                os.environ["MODELSCOPE_CACHE"] = str(ROOT / "指导/模型缓存/modelscope")
                os.environ["HF_HOME"] = str(ROOT / "指导/模型缓存/huggingface")
                import pytrio

                if "llm_timeout_seconds" in config:
                    pytrio.configure(
                        sampling_timeout_seconds=config["llm_timeout_seconds"],
                        sampling_first_chunk_timeout_seconds=config["llm_timeout_seconds"],
                        sampling_chunk_idle_timeout_seconds=config["llm_timeout_seconds"],
                        sampling_retry_timeout_seconds=30,
                        actor_capacity_wait_timeout_seconds=120)
                self.client = pytrio.ServiceClient().create_sampling_client(
                    base_model=model, model_path=model_path)
                self.tokenizer = self.client.get_tokenizer()
            except Exception as exc:
                self.initialization_error = f"{type(exc).__name__}: {exc}"

    def decide(self, observation):
        audit = {"source": "fallback_rule", "input_tokens": 0, "output_tokens": 0,
                 "proposal_valid": False, "fallback_reason": None}
        def fallback():
            return rule_action(observation, self.config)
        if observation["terminal"]:
            return fallback(), {**audit, "source": "terminal_guard", "fallback_reason": None}
        if self.initialization_error or self.calls >= self.config["max_llm_calls"]:
            audit["fallback_reason"] = self.initialization_error or "llm_call_budget_exhausted"
            return fallback(), audit
        try:
            import pytrio

            prompt = build_prompt(observation, self.config)
            if self.before_request:
                self.before_request()
            self.calls += 1  # Never refund failed or invalid requests.
            future = self.client.sample(
                prompt=pytrio.ModelInput.from_ints(tokens=prompt_tokens(prompt, self.tokenizer)),
                num_samples=1,
                sampling_params=pytrio.SamplingParams(
                    max_tokens=self.config["generation_max_tokens"], temperature=0, seed=42),
            )
            if "llm_timeout_seconds" in self.config:
                try:
                    response = future.result(timeout=self.config["llm_timeout_seconds"] + 10)
                except TimeoutError:
                    future.cancel()
                    raise
            else:
                response = future.result()
            sequence = response.sequences[0]
            decoded = self.tokenizer.decode(sequence.tokens, skip_special_tokens=True)
            audit.update({"raw_response": sequence.text, "decoded_response": decoded,
                          "tokens": sequence.tokens, "input_tokens": response.input_tokens,
                          "output_tokens": response.output_tokens})
            proposal = validate_action(parse_decision(decoded), observation, self.config)
            audit.update(source="llm", proposal_valid=True)
            return proposal, audit
        except Exception as exc:
            audit["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return fallback(), audit
