# scripts/test_pipeline.py
import sys
import os
sys.path.insert(0, '.')

from app.model    import load_model
from app.analyzer import init_analyzer, analyze

# ================================================
# SETUP
# ================================================
print("="*60)
print("FULL PIPELINE TEST — CCPA Compliance System")
print("="*60)

load_model()
init_analyzer()

# ================================================
# TEST CASES
# ================================================
test_cases = [
    # (prompt, expected_harmful, description)
    (
        "We are selling our customers personal information to "
        "third-party data brokers without informing them.",
        True,
        "Selling data without consent"
    ),
    (
        "A customer asked us to delete their data but we are "
        "ignoring their request and keeping all records.",
        True,
        "Ignoring deletion request"
    ),
    (
        "We charge customers who opted out of data selling "
        "a higher price for the same service.",
        True,
        "Discriminatory pricing"
    ),
    (
        "Our website has no privacy policy and we never tell "
        "users what data we collect from them.",
        True,
        "No privacy notice"
    ),
    (
        "We share users sensitive health and financial data "
        "with third parties without any restriction.",
        True,
        "Misuse of sensitive data"
    ),
    (
        "Our website has no Do Not Sell My Personal Information link.",
        True,
        "Missing opt-out link"
    ),
    (
        "Our company provides a clear privacy policy and allows "
        "customers to opt out of data selling at any time.",
        False,
        "Compliant — clear privacy policy"
    ),
    (
        "We deleted all personal data within 45 days after "
        "receiving the consumer verified request.",
        False,
        "Compliant — proper deletion"
    ),
    (
        "Can we schedule a team meeting for next Monday?",
        False,
        "Unrelated to CCPA"
    ),
    (
        "We provide equal service and pricing to all customers "
        "regardless of whether they exercise their privacy rights.",
        False,
        "Compliant — non-discriminatory"
    ),
]

# ================================================
# RUN TESTS
# ================================================
print(f"\nRunning {len(test_cases)} test cases...\n")

correct  = 0
harmful_correct = 0
safe_correct    = 0
total_harmful   = len([t for t in test_cases if t[1]])
total_safe      = len([t for t in test_cases if not t[1]])

for prompt, expected_harmful, desc in test_cases:
    result          = analyze(prompt)
    got_harmful     = result["harmful"]
    got_articles    = result["articles"]
    is_correct      = got_harmful == expected_harmful

    if is_correct:
        correct += 1
        if expected_harmful:
            harmful_correct += 1
        else:
            safe_correct += 1

    status = "✅" if is_correct else "❌"
    kind   = "HARMFUL" if expected_harmful else "SAFE"

    print(f"{status} [{kind}] {desc}")
    print(f"   Prompt:   '{prompt[:65]}'")
    print(f"   Expected: harmful={expected_harmful}")
    print(f"   Got:      harmful={got_harmful}, articles={got_articles}")
    print()

# ================================================
# RESULTS
# ================================================
print("="*60)
print("RESULTS")
print("="*60)
print(f"  Overall:          {correct}/{len(test_cases)}")
print(f"  Harmful cases:    {harmful_correct}/{total_harmful}")
print(f"  Safe cases:       {safe_correct}/{total_safe}")

pct = correct / len(test_cases) * 100
if pct >= 90:
    grade = "🏆 EXCELLENT — Competition ready!"
elif pct >= 70:
    grade = "✅ GOOD — Minor issues to fix"
elif pct >= 50:
    grade = "⚠️  NEEDS WORK"
else:
    grade = "❌ POOR — Check API and prompts"

print(f"  Score:            {pct:.0f}% — {grade}")
print("="*60)