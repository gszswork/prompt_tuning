"""
Unified LLM API module with caching infrastructure.

This module provides a centralized interface for making LLM API calls with:
- Automatic caching of responses
- Retry logic with exponential backoff
- Consistent error handling
- Support for multiple message formats
- Support for both OpenAI API and Hugging Face local models
"""

import hashlib
import json
import logging
import os
import time
from typing import Optional, Union, List, Dict
from openai import OpenAI

# Default cache directory
DEFAULT_CACHE_DIR = "./llm_cache"

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


class HuggingFaceBackend:
    """
    Backend for Hugging Face local models.

    Supports loading and running inference with transformers models locally.
    Optimized for multi-GPU setups (e.g., 4x A100).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
    ):
        """
        Initialize Hugging Face model backend.

        Args:
            model_name: Model name or path from Hugging Face Hub
            device: Device to load model on ("auto", "cuda", "cuda:0", etc.)
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
            torch_dtype: Torch dtype ("auto", "float16", "bfloat16", "float32")
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "torch and transformers libraries are required for Hugging Face backend. "
                "Install with: pip install torch transformers accelerate"
            ) from e

        self.model_name = model_name
        self.device = device

        logging.info(f"Loading Hugging Face model: {model_name}")

        # Determine torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        actual_dtype = dtype_map.get(torch_dtype, "auto")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        model_kwargs = {
            "pretrained_model_name_or_path": model_name,
            "trust_remote_code": True,
            "torch_dtype": actual_dtype,
        }

        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"
        elif load_in_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["device_map"] = "auto"
        elif device == "auto":
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Move to device if not using device_map
        if device != "auto" and not load_in_8bit and not load_in_4bit:
            self.model = self.model.to(device)

        self.model.eval()
        logging.info(f"Model loaded successfully on {self.device}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        Generate response from messages.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding

        Returns:
            Generated text response
        """
        import torch

        # Format messages into prompt using chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logging.warning(f"Failed to apply chat template: {e}, falling back to simple format")
                prompt = self._format_messages_simple(messages)
        else:
            prompt = self._format_messages_simple(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        )

        # Move to device
        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def _format_messages_simple(self, messages: List[Dict[str, str]]) -> str:
        """
        Simple fallback message formatting.

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted prompt string
        """
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        formatted.append("Assistant:")
        return "\n\n".join(formatted)


class LLMClient:
    """
    Unified LLM client with caching and retry logic.

    Supports both OpenAI API and Hugging Face local models.

    Usage (OpenAI):
        client = LLMClient(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        response = client.chat("Hello, how are you?", model="gpt-4o-mini")

    Usage (Hugging Face):
        client = LLMClient(
            provider="huggingface",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            device="auto",
            torch_dtype="bfloat16"
        )
        response = client.chat("Hello, how are you?")
    """

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        cache_dir: str = DEFAULT_CACHE_DIR,
        enable_cache: bool = True,
        default_model: str = "gpt-4o-mini",
        max_retries: int = 3,
        # Hugging Face specific parameters
        model_name: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        """
        Initialize the LLM client.

        Args:
            provider: Provider to use ("openai" or "huggingface")
            api_key: OpenAI API key (required for OpenAI provider)
            base_url: Base URL for the API endpoint (OpenAI only)
            cache_dir: Directory to store cached responses
            enable_cache: Whether to enable caching
            default_model: Default model to use if not specified (OpenAI only)
            max_retries: Maximum number of retry attempts
            model_name: Model name or path for Hugging Face (required for HF)
            device: Device to load model on (HF only)
            load_in_8bit: Load model in 8-bit precision (HF only)
            load_in_4bit: Load model in 4-bit precision (HF only)
            torch_dtype: Torch dtype for model (HF only)
            max_new_tokens: Max tokens to generate (HF only)
            temperature: Sampling temperature (HF only)
            top_p: Nucleus sampling parameter (HF only)
        """
        self.provider = provider.lower()
        self.cache_dir = cache_dir
        self.enable_cache = enable_cache
        self.default_model = default_model
        self.max_retries = max_retries

        # Initialize the appropriate backend
        if self.provider == "openai":
            if api_key is None:
                raise ValueError("api_key is required for OpenAI provider")
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.backend_type = "openai"
        elif self.provider == "huggingface":
            if model_name is None:
                raise ValueError("model_name is required for Hugging Face provider")
            self.hf_backend = HuggingFaceBackend(
                model_name=model_name,
                device=device,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                torch_dtype=torch_dtype,
            )
            self.backend_type = "huggingface"
            self.default_model = model_name
            # Store generation parameters
            self.max_new_tokens = max_new_tokens
            self.temperature = temperature
            self.top_p = top_p
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'huggingface'")

        # Create cache directory if it doesn't exist
        if self.enable_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _get_cache_key(self, messages: List[Dict], model: str) -> str:
        """
        Generate a unique cache key based on messages and model.

        Args:
            messages: List of message dictionaries
            model: Model name

        Returns:
            MD5 hash as cache key
        """
        cache_data = {
            "messages": messages,
            "model": model
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Load cached response if it exists.

        Args:
            cache_key: Cache key

        Returns:
            Cached response or None if not found
        """
        if not self.enable_cache:
            return None

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('response')
            except Exception as e:
                logging.warning(f"Failed to load cache file {cache_file}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, response: str):
        """
        Save response to cache.

        Args:
            cache_key: Cache key
            response: Response to cache
        """
        if not self.enable_cache:
            return

        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'response': response,
                    'timestamp': time.time()
                }, f, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Failed to save cache file {cache_file}: {e}")

    def chat_with_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Make an API call with custom messages format.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to self.default_model)
            use_cache: Whether to use cache for this request

        Returns:
            Response content as string

        Example:
            messages = [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"}
            ]
            response = client.chat_with_messages(messages)
        """
        model = model or self.default_model

        # Check cache if enabled
        if use_cache and self.enable_cache:
            cache_key = self._get_cache_key(messages, model)
            cached_result = self._load_from_cache(cache_key)
            if cached_result is not None:
                # logging.info(f"Cache hit for query (key: {cache_key[:8]}...)")
                return cached_result

        # Route to appropriate backend
        if self.backend_type == "openai":
            result = self._call_openai(messages, model)
        elif self.backend_type == "huggingface":
            result = self._call_huggingface(messages)
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

        # Store result in cache
        if use_cache and self.enable_cache:
            self._save_to_cache(cache_key, result)
            logging.info(f"Cached new result to disk (key: {cache_key[:8]}...)")

        return result

    def _call_openai(self, messages: List[Dict[str, str]], model: str) -> str:
        """
        Make OpenAI API call with retry logic.

        Args:
            messages: List of message dictionaries
            model: Model to use

        Returns:
            Response content as string
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    messages=messages,
                    model=model,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.warning(f"OpenAI API attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

    def _call_huggingface(self, messages: List[Dict[str, str]]) -> str:
        """
        Make Hugging Face model inference call with retry logic.

        Args:
            messages: List of message dictionaries

        Returns:
            Response content as string
        """
        for attempt in range(self.max_retries):
            try:
                return self.hf_backend.generate(
                    messages=messages,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            except Exception as e:
                logging.warning(f"Hugging Face inference attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e

    def chat(
        self,
        content: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Simple chat interface with a single user message.

        Args:
            content: User message content
            model: Model to use (defaults to self.default_model)
            system_message: Optional system message
            use_cache: Whether to use cache for this request

        Returns:
            Response content as string

        Example:
            response = client.chat("What is 2+2?")
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": content})

        return self.chat_with_messages(messages, model=model, use_cache=use_cache)

    def chat_with_instruction(
        self,
        input_text: str,
        instruction: str,
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Chat with instruction and input (common pattern in the codebase).

        Args:
            input_text: Input/question text
            instruction: Instruction for the model
            model: Model to use (defaults to self.default_model)
            use_cache: Whether to use cache for this request

        Returns:
            Response content as string

        Example:
            response = client.chat_with_instruction(
                input_text="The cat is on the mat.",
                instruction="Translate this to French."
            )
        """
        content = f"{input_text}\n{instruction}"
        return self.chat(content, model=model, use_cache=use_cache)

    def extract_answer(
        self,
        llm_answer: str,
        extraction_instruction: str,
        model: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Extract a specific format of answer from an LLM response.

        Args:
            llm_answer: The LLM's answer to extract from
            extraction_instruction: Instructions for extraction
            model: Model to use (defaults to self.default_model)
            use_cache: Whether to use cache for this request

        Returns:
            Extracted answer

        Example:
            extracted = client.extract_answer(
                llm_answer="The answer is clearly 42 because...",
                extraction_instruction="Extract only the numeric answer."
            )
        """
        content = f"The explanation is: {llm_answer}\n{extraction_instruction}"
        return self.chat(content, model=model, use_cache=use_cache)


# Convenience functions for common extraction patterns

def extract_yes_no(
    client: LLMClient,
    llm_answer: str,
    model: Optional[str] = None,
) -> str:
    instruction = "Extract the answer with Yes/No. Ensure the answer is only one of Yes and No without any punctuation"
    return client.extract_answer(llm_answer, instruction, model=model)

def extract_true_false(
    client: LLMClient,
    llm_answer: str,
    model: Optional[str] = None,
) -> str:
    instruction = "Extract the answer with True/False. Ensure the answer is only one of True and False without any punctuation"
    return client.extract_answer(llm_answer, instruction, model=model)

def extract_numeric(
    client: LLMClient,
    llm_answer: str,
    model: Optional[str] = None,
) -> str:
    instruction = "You are extracting the numeric answer from  a solution of a Math problem. Give the Math answer in the shortest form possible that will still be correct. Ensure the answer is only a numeric number."
    return client.extract_answer(llm_answer, instruction, model=model)


def extract_letters_only(
    client: LLMClient,
    llm_answer: str,
    model: Optional[str] = None,
) -> str:
    instruction = "We are solving the task to extract a single string from a text. The LLM answer with explanation is: {llm_answer}\n Extract the final string answer from the answer. Ensure the answer is only the string returned without any units, spaces between chars, punctuation, or explanatory text."
    return client.extract_answer(llm_answer, instruction, model=model)


def extract_choice(
    client: LLMClient,
    llm_answer: str,
    model: Optional[str] = None,
) -> str:
    instruction = "Extract the answer with only one of (A), (B), (C), (D), (E), (F) without the option's content (only a bracket and a letter). Show the answer without any preparatory statements"
    return client.extract_answer(llm_answer, instruction, model=model)
