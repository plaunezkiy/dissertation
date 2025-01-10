from typing import Any, Dict, Iterator, List, Mapping, Optional
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from mistral_inference.generate import generate
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM


class MistralLLM(LLM):
    tokenizer: MistralTokenizer
    model: Transformer

    def __init__(self):
        tokenizer = MistralTokenizer.from_file("/models/M7B/tokenizer.model.v3")  # change to extracted tokenizer file
        model = Transformer.from_folder("/models/M7B")  # change to extracted model dir
        super().__init__(model=model, tokenizer=tokenizer) 

    def get_response(self, prompt: str) -> str:
        tokenizer = self.tokenizer
        model = self.model
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens
        out_tokens, _ = generate([tokens], model, max_tokens=1024, temperature=0.35, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        return result

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        return self.get_response(prompt)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "Mistral-7B-0.3",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "mistral-7b-0.3"
