# app/analyzer.py
from app.retriever import CCPARetriever
from app.model    import run_inference, parse_llm_json
import json
import re

retriever = None


def init_analyzer():
    global retriever
    retriever = CCPARetriever("data/")
    print("[Analyzer] Ready!")


SYSTEM_MESSAGE = """You are a strict CCPA (California Consumer Privacy Act) legal compliance analyzer.

Your task:
1. Read the CCPA sections provided below
2. Analyze the business practice described by the user
3. Determine if it violates any CCPA section
4. Respond with ONLY a JSON object — no explanation, no markdown

STRICT OUTPUT FORMAT:
{"harmful": true, "articles": ["Section 1798.120", "Section 1798.115"]}
or
{"harmful": false, "articles": []}

RULES:
- harmful must be boolean true or false — never a string
- articles must always be a list — never null
- articles must be [] when harmful is false
- articles must have at least one entry when harmful is true
- Only cite section numbers from the CCPA sections provided to you
- Never invent or guess section numbers
- If the practice is unrelated to data privacy, return harmful: false
- If in doubt whether something is a violation, lean toward harmful: true"""


EXPLAIN_SYSTEM_MESSAGE = """You are a CCPA legal expert providing detailed compliance analysis.
You will be given a business practice and the CCPA sections it violates.
Provide a structured analysis in the following EXACT JSON format:

{
  "harmful": true or false,
  "articles": ["Section 1798.xxx"],
  "reasoning": "2-3 sentence explanation of WHY this is a violation and which specific rule is broken",
  "consequences": ["consequence 1", "consequence 2", "consequence 3"],
  "improvements": ["improvement 1", "improvement 2", "improvement 3"],
  "message": "A friendly one-liner verdict"
}

If NOT harmful, use this format:
{
  "harmful": false,
  "articles": [],
  "reasoning": "",
  "consequences": [],
  "improvements": [],
  "message": "✅ No law violated! You are good to go. Keep maintaining these privacy standards!"
}

RULES:
- consequences: list real legal/financial penalties from CCPA
- improvements: list specific actionable steps to fix the violation
- reasoning: cite the specific section and what rule it breaks
- message: keep it short and human-friendly
- Return ONLY the JSON object — no markdown, no explanation outside JSON"""


def build_messages(user_prompt: str, context: str) -> list:
    user_content = f"""RELEVANT CCPA SECTIONS:
{context}

BUSINESS PRACTICE TO ANALYZE:
{user_prompt}

Respond with ONLY the JSON object. No explanation. No markdown. No extra text."""

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user",   "content": user_content}
    ]


def build_explain_messages(user_prompt: str, context: str, articles: list) -> list:
    articles_str = ", ".join(articles) if articles else "none"

    user_content = f"""RELEVANT CCPA SECTIONS:
{context}

BUSINESS PRACTICE:
{user_prompt}

VIOLATIONS DETECTED: {articles_str}

Provide the full structured JSON analysis including reasoning, consequences, and improvements."""

    return [
        {"role": "system", "content": EXPLAIN_SYSTEM_MESSAGE},
        {"role": "user",   "content": user_content}
    ]


def analyze(prompt: str) -> dict:
    """
    Standard pipeline — returns only harmful + articles.
    Used by /analyze endpoint (hackathon grading).
    """
    global retriever

    results = retriever.retrieve(prompt, top_k=5)

    context = ""
    for r in results:
        context += f"\n{'='*40}\n"
        context += f"SECTION {r['section_number']} — {r['title']}\n"
        context += f"{'='*40}\n"
        text = r['full_text']
        if len(text) > 1200:
            text = text[:1200] + "\n... [section continues]"
        context += text + "\n"

    messages   = build_messages(prompt, context)
    raw_output = run_inference(messages, max_new_tokens=200)
    print(f"[Analyzer] Raw output: {raw_output}")

    result = parse_llm_json(raw_output)

    if result.get("harmful") and result.get("articles"):
        result["articles"] = retriever.validate_citations(result["articles"])
        if not result["articles"]:
            result["articles"] = [f"Section {results[0]['section_number']}"]

    if not result.get("harmful"):
        result["harmful"]  = False
        result["articles"] = []
    else:
        result["harmful"] = True

    return result


def analyze_with_explanation(prompt: str) -> dict:
    """
    Extended pipeline — returns full explanation.
    Used by /explain endpoint (human-readable rich response).
    """
    global retriever

    # Step 1: Run standard analysis first
    base_result = analyze(prompt)
    harmful     = base_result["harmful"]
    articles    = base_result["articles"]

    # Step 2: If not harmful return friendly message immediately
    if not harmful:
        return {
            "harmful":      False,
            "articles":     [],
            "reasoning":    "",
            "consequences": [],
            "improvements": [],
            "message":      "✅ No law violated! You are good to go. Keep maintaining these privacy-respecting standards and your business stays fully compliant!"
        }

    # Step 3: Get full explanation from LLM
    results = retriever.retrieve(prompt, top_k=5)

    context = ""
    for r in results:
        context += f"\n{'='*40}\n"
        context += f"SECTION {r['section_number']} — {r['title']}\n"
        context += f"{'='*40}\n"
        text = r['full_text']
        if len(text) > 1200:
            text = text[:1200] + "\n... [section continues]"
        context += text + "\n"

    messages   = build_explain_messages(prompt, context, articles)
    raw_output = run_inference(messages, max_new_tokens=600)
    print(f"[Analyzer] Explain raw output: {raw_output[:300]}")

    # Step 4: Parse rich response
    raw_output = re.sub(r'```json\s*', '', raw_output)
    raw_output = re.sub(r'```\s*',     '', raw_output)

    try:
        rich_result = json.loads(raw_output.strip())
    except:
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                rich_result = json.loads(match.group())
            except:
                rich_result = {}
        else:
            rich_result = {}

    # Step 5: Build final response — fallback gracefully if parsing failed
    return {
        "harmful":  harmful,
        "articles": articles,

        "reasoning": rich_result.get("reasoning") or
            f"This practice violates {', '.join(articles)} of the CCPA. "
            f"The described business behavior conflicts with consumer privacy rights "
            f"established under California law.",

        "consequences": rich_result.get("consequences") or [
            "Civil penalty of up to $2,500 per unintentional violation",
            "Civil penalty of up to $7,500 per intentional violation",
            "Private right of action by consumers — statutory damages $100-$750 per consumer",
            "Attorney General enforcement action",
            "Reputational damage and loss of consumer trust"
        ],

        "improvements": rich_result.get("improvements") or [
            "Immediately audit current data practices against CCPA requirements",
            "Update privacy policy to disclose all data collection and sharing",
            "Implement a consumer request portal for deletion, access, and opt-out",
            "Train staff on CCPA compliance requirements",
            "Consult a privacy attorney to review data processing agreements"
        ],

        "message": rich_result.get("message") or
            f"⚠️ CCPA Violation Detected! Immediate action required to avoid penalties."
    }