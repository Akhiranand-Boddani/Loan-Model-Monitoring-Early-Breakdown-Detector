FEATURE_DRIFT_PROMPT = '''
You are an expert ML monitoring analyst. Analyze the following drift diagnostics and produce a structured JSON response.

Input:
{{

	"numeric_table": """{numeric_table}""",
	"categorical_table": """{categorical_table}""",
	"prediction_summary": """{prediction_summary}""",
	"performance_summary": """{performance_summary}"""
}}
Requirements:
- Provide a short Chain-of-Thought as a list of 3-6 brief steps (strings) explaining how you interpret the signals.
- Provide a single-sentence `conclusion` summarizing model health.
- Provide `recommendations` as a JSON array (max 4 items) with short actionable steps.

Output JSON schema (exact keys required):
{{
	"chain_of_thought": ["step 1", "step 2", ...],
	"conclusion": "...",
	"recommendations": ["action 1", "action 2"]
}}

Respond ONLY with valid JSON following the schema above. Do not include any extra text.
'''

EXPLAINATION_PREFIX = "Return JSON with keys: chain_of_thought, conclusion, recommendations."
