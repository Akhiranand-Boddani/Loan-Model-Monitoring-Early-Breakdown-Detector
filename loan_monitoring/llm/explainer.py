import subprocess
import shlex
import os
import requests
import os
import requests
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
# Prefer a lightweight local Ollama model by default. Allow override via OLLAMA_MODEL env var.
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gemma:2b")
# Default temperature for deterministic output (0.0 is deterministic). Can be overridden via env var.
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.0"))


class LocalLLM:
    def __init__(self, model_name=MODEL_NAME, device=None):
        """
        Load a local LLM using Hugging Face Transformers.
        This delays importing torch until an instance is created to avoid
        Streamlit's module inspection touching torch internals during app startup.
        """
        # Lazy import torch to avoid side-effects at module-import time
        try:
            import torch
        except Exception:
            torch = None

        # Detect device: CPU or GPU
        if device:
            self.device = device
        else:
            if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

        if AutoTokenizer is None or AutoModelForCausalLM is None:
            raise ImportError("transformers package is required for LocalLLM. Please install with 'pip install transformers'.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # set torch dtype only if torch is available
        torch_dtype = None
        if torch is not None:
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            torch_dtype=torch_dtype,
        )
        if torch is not None:
            self.model = self.model.to(self.device)

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
    # Build human-readable content; tolerate different metric key names
    num_table_lines = []
    for k, v in feature_summary.get("numeric", {}).items():
        # support nested 'metrics' dict or flat keys
        metrics = v.get("metrics", {}) if isinstance(v, dict) else {}
        psi = v.get("psi", metrics.get("psi", None))
        kl = v.get("kl_divergence", metrics.get("kl_divergence", None))
        w = v.get("wasserstein_normalized", metrics.get("wasserstein_normalized", None))
        parts = []
        if psi is not None:
            parts.append(f"psi={psi:.3f}")
        if kl is not None:
            parts.append(f"kl={kl:.3f}")
        if w is not None:
            parts.append(f"wasserstein={w:.3f}")
        num_table_lines.append(f"{k}: {', '.join(parts) if parts else 'no metrics'}")
    num_table = "\n".join(num_table_lines)

    cat_table_lines = []
    for k, v in feature_summary.get("categorical", {}).items():
        metrics = v.get("metrics", {}) if isinstance(v, dict) else {}
        # support multiple possible chi2 keys
        chi2 = v.get("chi2_stat", metrics.get("chi2_stat", None))
        if chi2 is None:
            chi2 = v.get("chi2_statistic", metrics.get("chi2_statistic", None))
        if chi2 is None:
            chi2 = v.get("chi2", metrics.get("chi2", None))
        kl = v.get("kl_divergence", metrics.get("kl_divergence", None))
        parts = []
        if chi2 is not None:
            parts.append(f"chi2={chi2:.3f}")
        if kl is not None:
            parts.append(f"kl={kl:.3f}")
        cat_table_lines.append(f"{k}: {', '.join(parts) if parts else 'no metrics'}")
    cat_table = "\n".join(cat_table_lines)

    def fmt(x):
        try:
            # numpy types also handled
            if isinstance(x, (int, float)):
                return f"{x:.3f}"
            import numpy as _np
            if isinstance(x, _np.generic):
                return f"{float(x):.3f}"
        except Exception:
            pass
        try:
            return str(x)
        except Exception:
            return repr(x)

    pred_table = "\n".join([
        f"{model}: mean_proba={fmt(stats.get('mean_proba'))}, entropy={fmt(stats.get('entropy_mean'))}"
        for model, stats in (prediction_summary or {}).items()
    ])

    perf_table = ""
    if performance_summary:
        perf_table = "\n".join([f"{m}: {fmt(x)}" for m, x in performance_summary.items()])

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
    # Preferred order (to avoid loading large local transformer models):
    # 1) LangChain + Ollama (uses local Ollama server/models like gemma)
    # 2) Ollama CLI/Python fallback
    # 3) Gemini API (if configured)
    # 4) Local transformers only if explicitly forced via env var

    prompt_text = format_for_local_llm(feature_summary, prediction_summary, performance_summary)

    # 1) LangChain + Ollama wrapper (preferred - uses lightweight local ollama models)
    try:
        lc = try_langchain_ollama(prompt=prompt_text, model_name=model_name)
        if lc:
            return lc
    except Exception:
        pass

    # 2) Try Ollama (Python package or CLI) if available
    try:
        ollama_out = try_ollama(model_name=model_name, prompt=prompt_text)
        if ollama_out:
            return ollama_out
    except Exception:
        pass

    # 3) Try Gemini (user-provided API) via env vars
    try:
        gem = try_gemini(prompt=prompt_text, model_name=model_name)
        if gem:
            return gem
    except Exception:
        pass

    # 4) Only use bulky local transformers if explicitly requested
    if os.environ.get("FORCE_LOCAL_LLM", "0") == "1":
        try:
            llm = LocalLLM(model_name=model_name)
            explanation = llm.generate(prompt_text)
            return explanation
        except Exception:
            pass

    # Final fallback: deterministic transformer-free explanation
    return fallback_explain(
        feature_summary=feature_summary,
        prediction_summary=prediction_summary,
        performance_summary=performance_summary
    )


def try_ollama(model_name: str, prompt: str, timeout: int = 30) -> Optional[str]:
    """Attempt to run Ollama via Python package then CLI. Return generated text or None."""
    # Try Python package first
    try:
        import ollama
        try:
            # Best-effort: Ollama python API may expose a client or run function
            client = getattr(ollama, "Ollama", None)
            if client:
                c = client()
                # try common method names
                for method in ("create", "run", "predict"):
                    if hasattr(c, method):
                        fn = getattr(c, method)
                        try:
                            out = fn(model=model_name, prompt=prompt)
                            if isinstance(out, (str,)):
                                return out
                            if isinstance(out, dict) and "text" in out:
                                return out.get("text")
                            return str(out)
                        except Exception:
                            continue
        except Exception:
            pass
    except ImportError:
        pass

    # Fallback to CLI: `ollama run <model> --prompt '<prompt>'`
    try:
        # Prefer to detect local installed models first
        models = []
        for list_cmd in (["ollama", "list"], ["ollama", "ls"], ["ollama", "images"]):
            try:
                p = subprocess.run(list_cmd, capture_output=True, text=True, timeout=5)
                if p.returncode == 0 and p.stdout:
                    out = p.stdout.strip().splitlines()
                    # Parse lines: skip headers and take first token per line
                    for line in out:
                        line = line.strip()
                        if not line:
                            continue
                        # skip header lines
                        if any(h in line.lower() for h in ("name", "image", "model", "tags")) and not line.split()[0].startswith("/"):
                            # could be header; try next
                            continue
                        tok = line.split()[0]
                        if tok and tok not in models:
                            models.append(tok)
            except Exception:
                continue

        chosen = model_name
        # prefer explicit env override
        env_model = os.environ.get("OLLAMA_MODEL")
        if env_model:
            chosen = env_model
        elif models:
            # if requested model exists locally, use it; otherwise prefer gemma:2b if present
            if model_name in models:
                chosen = model_name
            elif "gemma:2b" in models:
                chosen = "gemma:2b"
            else:
                chosen = models[0]

        temp = float(os.environ.get("OLLAMA_TEMPERATURE", OLLAMA_TEMPERATURE))
        cmd = ["ollama", "run", chosen, "--prompt", prompt, "--temperature", str(temp)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0 and proc.stdout:
            return proc.stdout.strip()
    except Exception:
        pass

    return None


def try_langchain_ollama(prompt: str, model_name: str = MODEL_NAME, timeout: int = 30) -> Optional[str]:
    """Attempt to use LangChain's Ollama LLM wrapper if available.

    This is best-effort and uses a local import to avoid hard dependency at module import time.
    """
    try:
        # lazy import to avoid adding langchain as a hard dependency at import time
        from langchain.llms import Ollama
    except Exception:
        return None

    try:
        # Allow env override for model or use provided model_name
        chosen = os.environ.get("OLLAMA_MODEL", model_name)
        temp = float(os.environ.get("OLLAMA_TEMPERATURE", OLLAMA_TEMPERATURE))
        try:
            client = Ollama(model=chosen, temperature=temp)
            out = client(prompt)
        except TypeError:
            # older LangChain Ollama wrapper may not accept temperature argument
            client = Ollama(model=chosen)
            try:
                out = client(prompt, temperature=temp)
            except TypeError:
                out = client(prompt)

        if isinstance(out, str):
            return out
        # Some versions return a dict-like object with 'text'
        if hasattr(out, "get"):
            return out.get("text") or out.get("result") or str(out)
        return str(out)
    except Exception:
        return None


def try_gemini(prompt: str, model_name: str = MODEL_NAME, timeout: int = 30) -> Optional[str]:
    """Attempt to call a Gemini-compatible API endpoint using env vars.

    Expects environment variables:
      - GEMINI_API_URL: full URL to POST requests to
      - GEMINI_API_KEY: Bearer API key

    This is a best-effort adapter that supports multiple response shapes.
    """
    url = os.environ.get("GEMINI_API_URL")
    key = os.environ.get("GEMINI_API_KEY")
    if not url or not key:
        return None

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_output_tokens": 256
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            return None
        j = resp.json()
        # try common shapes
        # Vertex AI-style: {'candidates':[{'content':'...'}]}
        if isinstance(j, dict):
            if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                return j["candidates"][0].get("content") or j["candidates"][0].get("text")
            if "output" in j and isinstance(j["output"], list) and j["output"]:
                # some APIs return list of dicts with 'content' fields
                first = j["output"][0]
                if isinstance(first, dict):
                    return first.get("content") or first.get("text")
            # openai-compatible: {'choices':[{'text': '...'}]}
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                return j["choices"][0].get("text")
            # fallback: try top-level 'text' or 'content'
            if "text" in j:
                return j.get("text")
            if "content" in j:
                return j.get("content")
        # else return raw text
        return resp.text.strip()
    except Exception:
        return None


def _normalize_feature_summary(feature_summary: dict):
    """Convert various feature_summary shapes into dicts for numeric/categorical."""
    numeric = {}
    categorical = {}
    # If already grouped
    if isinstance(feature_summary, dict) and ("numeric" in feature_summary or "categorical" in feature_summary):
        numeric = feature_summary.get("numeric", {})
        categorical = feature_summary.get("categorical", {})
        return numeric, categorical

    # Otherwise, feature_summary is likely {feature: {metrics..., feature_type: 'numeric'}}
    for feat, info in (feature_summary or {}).items():
        ftype = info.get("feature_type") or info.get("type")
        if ftype == "numeric":
            numeric[feat] = info.get("metrics", {})
        else:
            categorical[feat] = info.get("metrics", {})
    return numeric, categorical


def fallback_explain(feature_summary: dict, prediction_summary: dict, performance_summary: Optional[dict] = None) -> str:
    """Create a human-friendly explanation without requiring transformers."""
    numeric, categorical = _normalize_feature_summary(feature_summary)

    lines = []
    lines.append("Automated Drift Summary (rule-based):")

    # Top numeric drifts by PSI/KL/Wasserstein if available
    if numeric:
        # compute a simple score for sort: prefer psi, then kl, then wasserstein
        scored = []
        for feat, m in numeric.items():
            psi = m.get("psi", 0) or m.get("metrics", {}).get("psi", 0)
            kl = m.get("kl_divergence", 0) or m.get("metrics", {}).get("kl_divergence", 0)
            w = m.get("wasserstein_normalized", 0) or m.get("metrics", {}).get("wasserstein_normalized", 0)
            score = float(psi) * 2.0 + float(kl) * 1.5 + float(w)
            scored.append((score, feat, psi, kl, w))
        scored.sort(reverse=True)
        topn = scored[:8]
        lines.append(f"Top numeric features by drift: {', '.join([f for _, f, *_ in topn])}")
    else:
        lines.append("No numeric features analyzed.")

    # Categorical drifts
    if categorical:
        cat_scored = []
        for feat, m in categorical.items():
            chi2_p = m.get("chi2_p_value", m.get("chi2_p", 1.0)) or 1.0
            kl = m.get("kl_divergence", 0) or m.get("metrics", {}).get("kl_divergence", 0)
            score = (1.0 - float(chi2_p)) * 2.0 + float(kl)
            cat_scored.append((score, feat, chi2_p, kl))
        cat_scored.sort(reverse=True)
        topc = cat_scored[:8]
        lines.append(f"Top categorical features by drift: {', '.join([f for _, f, *_ in topc])}")
    else:
        lines.append("No categorical features analyzed.")

    # Prediction summary
    if prediction_summary:
        for model, stats in prediction_summary.items():
            lines.append(f"Model '{model}': mean_proba={stats.get('mean_proba', 0):.3f}, entropy={stats.get('entropy_mean', 0):.3f}")
    else:
        lines.append("No prediction summary available.")

    # Performance summary and recommendations
    if performance_summary:
        for m, metrics in performance_summary.items():
            if isinstance(metrics, dict):
                lines.append(f"Performance for {m}: accuracy={metrics.get('accuracy', 0):.3f}, recall={metrics.get('recall', 0):.3f}, roc_auc={metrics.get('roc_auc', 0):.3f}")
                if metrics.get('recall', 0) < 0.5:
                    lines.append(f"Recommendation: {m} has low recall; consider re-labeling recent data and retraining.")
            else:
                lines.append(f"Performance for {m}: {metrics}")
    else:
        lines.append("No performance labels available to evaluate model quality.")

    lines.append("Recommended next steps:")
    lines.append("- Prioritize labeling of recent data in flagged segments.")
    lines.append("- Retrain models including recent samples and re-evaluate metrics.")
    lines.append("- Investigate feature drift in top-reported features (see dashboard tables).")

    return "\n".join(lines)
