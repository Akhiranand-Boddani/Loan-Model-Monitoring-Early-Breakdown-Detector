import torch
# Make sure to install 'transformers' package: pip install transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    # You must install 'transformers' for LLM functionality
from loan_monitoring.llm.prompts import FEATURE_DRIFT_PROMPT
from typing import Optional

# Choose a local model that can be run on CPU for testing
# GPT-J-6B is a common open-source choice for local setups
MODEL_NAME = "EleutherAI/gpt-j-6B"


class LocalLLM:
    def __init__(self, model_name=MODEL_NAME, device=None):
        """
        Load a local LLM using Hugging Face Transformers.
        """
        # Detect device: CPU or GPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers package is required for LocalLLM. Please install with 'pip install transformers'.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Generate text for a prompt. Returns the model's output.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return text


# ---------------------------------------------
# Functions used by streamlit_app.py
# ---------------------------------------------

def format_for_local_llm(
    feature_summary: dict,
    prediction_summary: dict,
    performance_summary: Optional[dict] = None
) -> str:
    """
    Create a prompt string for the local LLM.
    """
    # Build human-readable content
    num_table = "\n".join(
        [f"{k}: psi={v['psi']:.3f}" for k, v in feature_summary.get("numeric", {}).items()]
    )

    cat_table = "\n".join(
        [f"{k}: chi2={v['chi2_stat']:.3f}" for k, v in feature_summary.get("categorical", {}).items()]
    )

    pred_table = "\n".join(
        [f"{model}: mean_proba={stats['mean_proba']:.3f}, entropy={stats['entropy_mean']:.3f}"
         for model, stats in prediction_summary.items()]
    )

    perf_table = ""
    if performance_summary:
        perf_table = "\n".join([f"{m}: {x:.3f}" for m, x in performance_summary.items()])

    prompt = FEATURE_DRIFT_PROMPT.format(
        numeric_table=num_table,
        categorical_table=cat_table,
        prediction_summary=pred_table,
        performance_summary=perf_table
    )

    return prompt


def explain_drift_locally(
    feature_summary: dict,
    prediction_summary: dict,
    performance_summary: Optional[dict] = None,
    model_name: str = MODEL_NAME
) -> str:
    """
    Use the local LLM to generate an explanation.
    """
    llm = LocalLLM(model_name=model_name)

    prompt = format_for_local_llm(
        feature_summary=feature_summary,
        prediction_summary=prediction_summary,
        performance_summary=performance_summary
    )

    explanation = llm.generate(prompt)
    return explanation
