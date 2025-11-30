from threading import Lock
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import os

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

from helm.common.cache import CacheConfig
from helm.common.gpu_utils import get_torch_device_name
from helm.common.hierarchical_logger import hexception, hlog, htrack_block
from helm.common.media_object import TEXT_TYPE
from helm.common.request import Request, RequestResult, GeneratedOutput, Token
from helm.common.request import wrap_request_time
from helm.clients.client import CachingClient, generate_uid_for_multimodal_prompt


@dataclass(frozen=True)
class LoadedModelProcessor:
    model: Any
    processor: AutoProcessor


# Global cache for all models - now dynamically populated
_models_lock: Lock = Lock()
_models: Dict[str, Optional[LoadedModelProcessor]] = {}


class Qwen2VLMClient(CachingClient):
    def __init__(
        self,
        cache_config: CacheConfig,
        pretrained_model_name_or_path: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        **_: Any,
    ):
        """
        Modified constructor to allow custom model paths and tokenizer names.

        Args:
            pretrained_model_name_or_path (str): local path or HF name.
            tokenizer_name (str): tokenizer to load (defaults to same as model).
        """
        super().__init__(cache_config=cache_config)

        self._device: str = get_torch_device_name()

        # Store configuration
        self._override_model_path = pretrained_model_name_or_path
        self._override_tokenizer_name = tokenizer_name
        self._trust_remote_code = trust_remote_code
        self._device_map = device_map

    def _get_default_model_name(self, helm_model_name: str) -> str:
        mapping = {
            "qwen2-vl-7b-instruct": "Qwen/Qwen2-VL-7B-Instruct",
            "qwen2-vl-72b-instruct": "Qwen/Qwen2-VL-72B-Instruct",
            "qwen2.5-vl-3b-instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
            "qwen2.5-vl-7b-instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen2.5-vl-32b-instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
            "qwen2.5-vl-72b-instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
        }
        if helm_model_name not in mapping:
            raise ValueError(f"Unhandled model name: {helm_model_name}")

        return mapping[helm_model_name]

    def _resolve_model_and_tokenizer(self, helm_model_name: str):
        # If user specified a custom path/model, use it.
        if self._override_model_path:
            model_path = self._override_model_path
            # Check if path exists and provide better error message
            if os.path.exists(model_path):
                hlog(f"Model path exists: {model_path}")
#                # List contents for debugging
#                if os.path.isdir(model_path):
#                    hlog(f"Contents of model directory {model_path}:")
#                    try:
#                        for item in os.listdir(model_path):
#                            hlog(f"  - {item}")
#                    except Exception as e:
#                        hlog(f"Could not list directory contents: {e}")
            else:
                hlog(f"Model path does not exist, treating as Hugging Face model ID: {model_path}")
        else:
            # Default to HF model
            model_path = self._get_default_model_name(helm_model_name)

        # Tokenizer defaults to same source as the model if not provided
        # If tokenizer_name is provided but doesn't exist as a path, it might be a Hugging Face name
        tokenizer_path = self._override_tokenizer_name or model_path

        return model_path, tokenizer_path

    def _get_model(self, helm_model_name: str) -> LoadedModelProcessor:
        from transformers import (
            Qwen2VLForConditionalGeneration,
            Qwen2_5_VLForConditionalGeneration,
            AutoModelForCausalLM,
        )

        global _models_lock, _models

        model_path, tokenizer_path = self._resolve_model_and_tokenizer(helm_model_name)

        # Use model_path as key for caching - this handles both HF models and local paths
        cache_key = model_path

        with _models_lock:
            loaded = _models.get(cache_key)
            if loaded is None:
                hlog(f"Loading model from {model_path} ...")
                hlog(f"Using tokenizer from {tokenizer_path} ...")

                # Determine model class
                is_qwen25 = ("2.5" in model_path.lower()) or any(ft_name in model_path for ft_name in [
                    "qwen2.5-vl-3b-instruct-ft",
                    "qwen2.5-vl-3b-instruct-ft2"
                ])

                model_class = (
                    Qwen2_5_VLForConditionalGeneration
                    if is_qwen25
                    else Qwen2VLForConditionalGeneration
                )

                hlog(f"Using model class: {model_class.__name__}")
                is_local_path = os.path.exists(model_path)

                try:
                    # Try loading without flash attention first for local models
                    model = model_class.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map=self._device_map,
                        trust_remote_code=self._trust_remote_code,
                        local_files_only=is_local_path,
                    ).eval()
                except Exception as e:
                    hlog(f"Failed to load with specific class, trying AutoModel... Error: {e}")
                    # Fall back to AutoModel
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map=self._device_map,
                        trust_remote_code=self._trust_remote_code,
                        local_files_only=is_local_path,
                    ).eval()

                # For tokenizer, don't check if path exists - let Hugging Face handle it
                # It could be a local path OR a Hugging Face model name
                try:
                    processor = AutoProcessor.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=self._trust_remote_code,
                        local_files_only=is_local_path,
                    )
                except Exception as e:
                    hlog(f"Failed to load processor from {tokenizer_path}, trying model path... Error: {e}")
                    # Fall back to using model path for processor
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=self._trust_remote_code,
                        local_files_only=is_local_path,
                    )

                loaded = LoadedModelProcessor(model=model, processor=processor)
                _models[cache_key] = loaded
                hlog(f"Successfully loaded local model: {cache_key}")

        return loaded

    def make_request(self, request: Request) -> RequestResult:
        assert request.multimodal_prompt is not None, "Multimodal prompt is required"

        # Build messages by collating all media objects into a single "user" message.
        message_content = []
        for media_object in request.multimodal_prompt.media_objects:
            if media_object.is_type("image") and media_object.location:
                message_content.append({"type": "image", "image": media_object.location})
            elif media_object.is_type(TEXT_TYPE):
                if media_object.text is None:
                    raise ValueError("MediaObject of text type has missing text field value")
                message_content.append({"type": "text", "text": media_object.text})
            else:
                raise ValueError(f"Unrecognized MediaObject type {media_object.type}")

        messages = [{"role": "user", "content": message_content}]

        generation_args = {
            "max_new_tokens": request.max_tokens,
        }

        # Add temperature and top_p if specified
        if request.temperature > 0:
            generation_args["temperature"] = request.temperature
        if request.top_p > 0:
            generation_args["top_p"] = request.top_p

        completions: List[GeneratedOutput] = []
        request_time: float = 0
        request_datetime: Optional[int] = None
        all_cached: bool = True

        with htrack_block(f"Generating for prompt: {request.multimodal_prompt.text}"):
            for completion_index in range(request.num_completions):
                try:

                    def do_it() -> Dict[str, Any]:
                        loaded = self._get_model(request.model_engine)
                        model = loaded.model
                        processor = loaded.processor

                        # Prepare text and vision inputs.
                        text = processor.apply_chat_template(   # type: ignore
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = processor(     # type: ignore
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(self._device)

                        with torch.no_grad():
                            generated_ids = model.generate(**inputs, **generation_args)

                        # Remove the input prefix from outputs.
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = processor.batch_decode(   # type: ignore
                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )
                        # For simplicity, split tokens by whitespace.
                        completion = output_text[0]
                        tokens = completion.split()
                        return {"output": (completion, tokens)}

                    cache_key = CachingClient.make_cache_key(
                        raw_request={
                            "completion_index": completion_index,
                            "model": request.model,
                            "prompt": generate_uid_for_multimodal_prompt(request.multimodal_prompt),
                            **generation_args,
                        },
                        request=request,
                    )
                    result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
                except RuntimeError as model_error:
                    hexception(model_error)
                    return RequestResult(
                        success=False,
                        cached=False,
                        error=str(model_error),
                        completions=[],
                        embedding=[],
                    )

                text_out, tokens = result["output"]
                completions.append(
                    GeneratedOutput(
                        text=text_out,
                        logprob=0,
                        tokens=[Token(text=str(token), logprob=0) for token in tokens],
                    )
                )
                hlog(f"Generated: {text_out}")
                request_time += result["request_time"]
                request_datetime = request_datetime or result.get("request_datetime")
                all_cached = all_cached and cached

        return RequestResult(
            success=True,
            cached=all_cached,
            request_time=request_time,
            request_datetime=request_datetime,
            completions=completions,
            embedding=[],
        )
