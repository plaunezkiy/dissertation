from typing import Any, Dict, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2TokenizerFast
from langchain_core.language_models.llms import LLM


class Qwen2_5(LLM):
    tokenizer: Qwen2TokenizerFast
    model: Qwen2ForCausalLM

    def __init__(self):
        model_path = "/models/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        super().__init__(model=model, tokenizer=tokenizer) 

    def get_response(self, prompt: str, temperature: float=0.15) -> str:
        tokenizer = self.tokenizer
        model = self.model
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
        outputs = model.generate(**inputs, max_length=400, temperature=temperature)
        result = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1]:])[0]
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
            "model_name": "Qwen2.5-0.5B-Instruct",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "qwen2.5-0.5B-Instruct"
