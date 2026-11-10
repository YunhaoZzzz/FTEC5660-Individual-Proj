import ast
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import re
import string
from copy import deepcopy

def _try_json_block(s):
    # Try json.loads first (handles valid JSON)
    try:
        data = json.loads(s)
    except Exception:
        # Try single-quote replacement
        try:
            data = json.loads(s.replace("'", '"'))
        except Exception:
            # Fallback: treat the whole thing as a Python expression (handles r"...", True/False, etc.)
            import warnings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    data = ast.literal_eval(s)
                    # Reject if result contains non-JSON types (e.g., Python sets)
                    try:
                        json.dumps(data)
                    except (TypeError, ValueError):
                        return None
            except Exception:
                return None

    if not isinstance(data, dict):
        return None
    if "name" not in data:
        return None

    if "arguments" not in data:
        if "parameters" in data:
            data["arguments"] = data["parameters"]
        else:
            data["arguments"] = {}

    if isinstance(data["arguments"], str):
        try:
            data["arguments"] = json.loads(data["arguments"])
        except Exception:
            data["arguments"] = {}

    if not isinstance(data["arguments"], dict):
        data["arguments"] = {}

    return data


def _parse_tool_call_from_text(text):
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    if "<tool_call>" in text:
        body = text.split("<tool_call>", 1)[1]
        body = body.split("</tool_call>", 1)[0].strip()
        parsed = _try_json_block(body)
        if parsed is not None:
            return parsed

    parsed = _try_json_block(text)
    if parsed is not None:
        return parsed

    # Try all JSON-like blocks from short to long (robust to reasoning text).
    for m in re.finditer(r"\{[\s\S]*?\}", text):
        parsed = _try_json_block(m.group(0))
        if parsed is not None:
            return parsed

    return None


def _normalize_generation_output(decoded):
    parsed = _parse_tool_call_from_text(decoded)
    if parsed is not None:
        return {
            "type": "tool",
            "tool_call_id": "".join(random.sample(string.ascii_letters + string.digits, 9)),
            "tool_name": parsed["name"],
            "arguments": parsed.get("arguments", {}),
            "raw_text": decoded,
        }
    return {"type": "content", "content": decoded, "raw_text": decoded}


class HFAgentBackend:
    def __init__(
        self,
        model_path,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        device="cuda",
        enable_thinking=None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(self.device).eval()

        self.generation_config = GenerationConfig.from_pretrained(model_path)
        if temperature is not None:
            self.generation_config.temperature = float(temperature)
        if top_p is not None:
            self.generation_config.top_p = float(top_p)
        if top_k is not None:
            self.generation_config.top_k = int(top_k)

    def _apply_chat_template(self, messages, tools=None, **kwargs):
        call_kwargs = dict(kwargs)
        if tools is not None:
            call_kwargs["tools"] = tools
        if self.enable_thinking is not None:
            call_kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            # Some tokenizers/templates do not support enable_thinking.
            call_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)

    def render_prompt(self, messages, tools):
        if tools:
            return self._apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )
        return self._apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        outs = []
        for idx, (messages, tools) in enumerate(zip(messages_batch, tools_batch)):
            prompt_text = self.render_prompt(messages, tools)
            pf = (prefills[idx] or "") if prefills else ""

            # Tokenize prompt + prefill together
            full_prompt = prompt_text + pf
            model_inputs = self.tokenizer(
                full_prompt, return_tensors="pt"
            ).to(self.device)

            gen_cfg = deepcopy(self.generation_config)
            output = self.model.generate(
                **model_inputs,
                generation_config=gen_cfg,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            decoded = self.tokenizer.decode(
                output[0][len(model_inputs["input_ids"][0]) :],
                skip_special_tokens=True,
            )
            # Prepend prefill so downstream parsing sees the full response
            full_text = pf + decoded
            out = _normalize_generation_output(full_text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = "length_or_eos"
            outs.append(out)
        return outs


class VLLMAgentBackend:
    def __init__(
        self,
        model_path,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        tensor_parallel_size=1,
        max_model_len=4096,
        dtype="bfloat16",
        enable_thinking=None,
    ):
        from transformers import AutoTokenizer, GenerationConfig
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.generation_config = GenerationConfig.from_pretrained(model_path)

        if temperature is not None:
            self.generation_config.temperature = float(temperature)
        if top_p is not None:
            self.generation_config.top_p = float(top_p)
        if top_k is not None:
            self.generation_config.top_k = int(top_k)

        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
        )

    def _sampling_params(self):
        kw = {"n": 1, "max_tokens": self.max_new_tokens}
        if getattr(self.generation_config, "temperature", None) is not None:
            kw["temperature"] = float(self.generation_config.temperature)
        if getattr(self.generation_config, "top_p", None) is not None:
            kw["top_p"] = float(self.generation_config.top_p)
        if getattr(self.generation_config, "top_k", None) is not None:
            kw["top_k"] = int(self.generation_config.top_k)
        if getattr(self.generation_config, "repetition_penalty", None) is not None:
            kw["repetition_penalty"] = float(self.generation_config.repetition_penalty)
        return self.SamplingParams(**kw)

    def _apply_chat_template(self, messages, tools=None, **kwargs):
        call_kwargs = dict(kwargs)
        if tools is not None:
            call_kwargs["tools"] = tools
        if self.enable_thinking is not None:
            call_kwargs["enable_thinking"] = self.enable_thinking
        try:
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)
        except TypeError:
            call_kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(messages, **call_kwargs)

    def render_prompt(self, messages, tools):
        if tools:
            return self._apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=tools,
            )
        return self._apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        prompts = [self.render_prompt(m, t) for m, t in zip(messages_batch, tools_batch)]
        if prefills:
            prompts = [p + (pf or "") for p, pf in zip(prompts, prefills)]
        params = self._sampling_params()
        raw = self.llm.generate(prompts=prompts, sampling_params=params)

        outs = []
        for idx, (prompt_text, r) in enumerate(zip(prompts, raw)):
            gen_text = r.outputs[0].text if r.outputs else ""
            # Prepend the prefill to the raw output so downstream parsing sees the full response
            pf = (prefills[idx] or "") if prefills else ""
            full_text = pf + gen_text
            out = _normalize_generation_output(full_text)
            out["prompt_text"] = prompt_text
            out["finish_reason"] = r.outputs[0].finish_reason if r.outputs else "empty"
            outs.append(out)
        return outs


class DeepSeekAPBackend:
    """OpenAI-compatible DeepSeek backend for tool-necessity evaluation."""

    def __init__(self, model_path, temperature=0.7, top_p=None,
                 max_new_tokens=512, base_url=None, max_workers=4):
        from openai import OpenAI

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY in the environment before running.")
        self.model_name = model_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.max_workers = max_workers
        self.base_url = base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)
        self.tokenizer = None

    def render_prompt(self, messages, tools=None):
        return json.dumps(messages, ensure_ascii=False)

    def _complete_one(self, messages, tools):
        request = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        if self.top_p is not None:
            request["top_p"] = self.top_p
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**request)
        message = response.choices[0].message
        content = message.content or ""
        prompt_text = self.render_prompt(messages, tools)

        if message.tool_calls:
            # The harness executes one tool at a time. Preserve one matching call
            # in the conversation so the next API request remains well formed.
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            api_message = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [{
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments or "{}",
                    },
                }],
            }
            return {
                "type": "tool",
                "tool_call_id": call.id,
                "tool_name": call.function.name,
                "arguments": arguments if isinstance(arguments, dict) else {},
                "content": content,
                "raw_text": content,
                "api_message": api_message,
                "prompt_text": prompt_text,
                "finish_reason": response.choices[0].finish_reason,
                "api_usage": {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                },
            }
        return {
            "type": "content",
            "content": content,
            "raw_text": content,
            "prompt_text": prompt_text,
            "finish_reason": response.choices[0].finish_reason,
            "api_usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
            },
        }

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        pairs = list(zip(messages_batch, tools_batch))
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda pair: self._complete_one(*pair), pairs))


class AgentModel:
    def __init__(
        self,
        model_path,
        backend="vllm",
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
        device="cuda",
        tensor_parallel_size=1,
        max_model_len=4096,
        vllm_dtype="bfloat16",
        enable_thinking=None,
        system_prompt_override=None,
        api_base_url=None,
        api_max_workers=4,
    ):
        from utils import get_system_prompt, detect_tool_format
        if system_prompt_override:
            self.system_prompt = system_prompt_override
        else:
            fmt = detect_tool_format(model_path)
            self.system_prompt = get_system_prompt(fmt)

        if backend == "deepseek":
            self.engine = DeepSeekAPBackend(
                model_path=model_path,
                temperature=temperature if temperature is not None else 0.7,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                base_url=api_base_url,
                max_workers=api_max_workers,
            )
            self.tokenizer = None
        elif backend == "hf":
            self.engine = HFAgentBackend(
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                device=device,
                enable_thinking=enable_thinking,
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        elif backend == "vllm":
            self.engine = VLLMAgentBackend(
                model_path=model_path,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                dtype=vllm_dtype,
                enable_thinking=enable_thinking,
            )
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def generate_batch(self, messages_batch, tools_batch, prefills=None):
        return self.engine.generate_batch(messages_batch, tools_batch, prefills=prefills)

    def generate(self, messages, tools, prefill=None):
        prefills = [prefill] if prefill is not None else None
        return self.generate_batch([messages], [tools], prefills=prefills)[0]

    def render_prompt(self, messages, tools):
        return self.engine.render_prompt(messages, tools)
