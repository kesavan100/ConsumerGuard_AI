"""
ConsumerGuard AI — Agent 1 (Complaint Analysis) + Agent 3 (Resolution)

Agent 1 — Complaint Analysis Agent
    Uses regex and keyword matching ONLY (no LLM).
    Fast, deterministic, zero API cost.
    Extracts: platform, issue_type, product_category, timeline, entities.

Agent 3 — Resolution Agent
    Uses Gemini API to generate the final user-facing response.
    Strictly grounded in retrieved policy chunks (no hallucination).
    Confidence level is computed by rule-based logic, not by Gemini.
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from consumerguard.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    HIGH_SIMILARITY_THRESHOLD,
    MEDIUM_SIMILARITY_THRESHOLD,
)


# ─── Agent 1: Complaint Analysis ─────────────────────────────────────────────

# Keywords for platform detection
AMAZON_KEYWORDS = [
    "amazon", "amazon.in", "prime", "alexa", "fulfillment by amazon", "fba",
    "amazon seller", "amazon pay", "amazon fresh",
]
FLIPKART_KEYWORDS = [
    "flipkart", "flipkart.com", "myntra", "jabong", "flipkart assured",
    "supermart", "2gud", "flipkart seller",
]

# Keywords for issue type detection (checked in order; first match wins)
ISSUE_KEYWORDS: Dict[str, List[str]] = {
    "Defective Product": [
        "defective", "faulty", "not working", "damaged", "broken", "malfunction",
        "doesn't work", "dead on arrival", "doa", "defect", "manufacturing defect",
    ],
    "Return Rejected": [
        "return rejected", "rejected return", "return denied", "denied return",
        "refused return", "return refused", "won't accept return", "return not accepted",
        "return request rejected", "unable to return",
    ],
    "Replacement Refused": [
        "replacement refused", "refused replacement", "replacement rejected",
        "replacement denied", "no replacement", "won't replace", "replacement request denied",
    ],
    "Refund Delayed": [
        "refund delayed", "refund not received", "no refund", "refund pending",
        "waiting for refund", "refund not processed", "refund issue", "money not returned",
        "money not refunded", "amount not credited",
    ],
    "Pickup Not Scheduled": [
        "pickup not scheduled", "pickup not arranged", "no pickup", "pickup failed",
        "pickup not done", "pickup not assigned", "pickup cancelled", "no pickup agent",
    ],
    "Third-party Seller Issue": [
        "third party", "third-party", "seller", "marketplace seller", "fulfilled by seller",
        "sold by", "vendor issue", "seller refused",
    ],
    "Cancellation Issue": [
        "cancelled", "cancellation", "order cancelled", "auto cancelled", "cancel",
        "order not cancelled", "cancellation not processed",
    ],
}

# Keywords for product category detection
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "Electronics": [
        "laptop", "mobile", "phone", "smartphone", "tablet", "computer", "tv", "television",
        "monitor", "headphone", "earphone", "camera", "charger", "battery", "keyboard",
        "mouse", "router", "smartwatch", "watch", "speaker", "audio", "electronics",
        "iphone", "samsung", "realme", "redmi", "oneplus", "ipad",
    ],
    "Fashion": [
        "shirt", "dress", "shoes", "footwear", "jeans", "trouser", "saree", "kurta",
        "t-shirt", "jacket", "bag", "wallet", "sunglasses", "clothing",
        "fashion", "apparel", "wear", "outfit", "sneakers",
    ],
    "Home & Furniture": [
        "refrigerator", "fridge", "washing machine", "ac", "air conditioner", "microwave",
        "oven", "mixer", "grinder", "fan", "iron", "cooler", "geyser", "water purifier",
        "vacuum cleaner", "dishwasher", "induction", "appliance", "sofa", "bed", "table",
        "chair", "mattress", "furniture",
    ],
    "Beauty & Personal Care": [
        "makeup", "perfume", "trimmer", "skincare", "shampoo", "cosmetics", "lotion", "cream",
    ],
    "Grocery & Food": [
        "snacks", "food", "beverage", "staples", "oil", "grocery", "groceries",
    ],
    "Books & Media": [
        "book", "novel", "cd", "dvd", "magazine",
    ],
    "Automotive": [
        "tyre", "helmet", "car accessory", "bike accessory",
    ],
    "Health & Fitness": [
        "supplements", "treadmill", "dumbbell", "protein", "fitness",
    ],
}

# Regex patterns for timeline extraction
TIMELINE_PATTERNS = [
    r"(\d+)\s*days?",
    r"(\d+)\s*weeks?",
    r"(\d+)\s*months?",
    r"(\d+)\s*hours?",
    r"within\s+(\d+)\s*days?",
    r"after\s+(\d+)\s*days?",
]


def analyze_complaint(complaint_text: str) -> Dict:
    """
    Agent 1: Extract structured information from raw complaint text.
    Uses ONLY regex and keyword matching — no LLM calls.

    Args:
        complaint_text: Raw complaint text from the user.

    Returns:
        {
            "platform": "Amazon" | "Flipkart" | "Unknown",
            "issue_type": str,
            "product_category": str,
            "timeline": str | None,
            "entities": List[str],
        }
    """
    text_lower = complaint_text.lower()

    # ── Platform Detection ──────────────────────────────────────────────────
    platform = "Unknown"
    amazon_score = sum(1 for kw in AMAZON_KEYWORDS if kw in text_lower)
    flipkart_score = sum(1 for kw in FLIPKART_KEYWORDS if kw in text_lower)

    if amazon_score > flipkart_score:
        platform = "Amazon"
    elif flipkart_score > amazon_score:
        platform = "Flipkart"
    elif amazon_score > 0 and amazon_score == flipkart_score:
        platform = "Amazon"  # tie-break: default to Amazon (more common query)

    # ── Issue Type Detection ────────────────────────────────────────────────
    issue_type = "General Complaint"
    for issue, keywords in ISSUE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            issue_type = issue
            break  # first match wins; order in ISSUE_KEYWORDS matters

    # ── Product Category Detection ──────────────────────────────────────────
    product_category = "Others"
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            product_category = category
            break

    # ── Timeline Extraction ─────────────────────────────────────────────────
    timeline = None
    for pattern in TIMELINE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            raw_number = match.group(1)
            # Determine the unit from the pattern itself
            if "week" in pattern:
                timeline = f"{raw_number} week(s)"
            elif "month" in pattern:
                timeline = f"{raw_number} month(s)"
            elif "hour" in pattern:
                timeline = f"{raw_number} hour(s)"
            else:
                timeline = f"{raw_number} day(s)"
            break

    # ── Entity Extraction ───────────────────────────────────────────────────
    # Extract capitalized words (potential product names, order IDs, etc.)
    entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", complaint_text)
    # Filter out common words that shouldn't count as entities
    stop_words = {
        "Amazon", "Flipkart", "I", "My", "The", "They", "He", "She",
        "We", "You", "It", "This", "That", "But", "And", "Or", "For",
    }
    entities = [e for e in entities if e not in stop_words]
    entities = list(dict.fromkeys(entities))[:5]  # deduplicate, limit to 5

    return {
        "platform": platform,
        "issue_type": issue_type,
        "product_category": product_category,
        "timeline": timeline,
        "entities": entities,
    }


# ─── Confidence Level (Rule-Based) ────────────────────────────────────────────

def compute_confidence(
    platform: str,
    best_similarity: float,
    chunk_count: int,
) -> Tuple[str, str]:
    """
    Compute a confidence level based on objective retrieval metrics.
    This is NOT generated by the LLM — it is fully deterministic.

    Rules:
    - HIGH:   similarity > 0.75
    - MEDIUM: similarity > 0.55
    - LOW:    otherwise

    Args:
        platform:        Detected platform ("Amazon", "Flipkart", or "Unknown").
        best_similarity: Highest similarity score from retrieval (0–1).
        chunk_count:     Number of chunks retrieved.

    Returns:
        Tuple of (Level, Reason)
    """
    if best_similarity > HIGH_SIMILARITY_THRESHOLD:
        return "HIGH", "High confidence because relevant platform policy and consumer law information were retrieved with strong similarity."
    elif best_similarity > MEDIUM_SIMILARITY_THRESHOLD:
        return "MEDIUM", "Medium confidence because product-specific policy information was retrieved, but may not perfectly match the scenario."
    else:
        return "LOW", "Low confidence because specific policy information was unavailable or a platform was not clearly detected."


# ─── Agent 3: Resolution Agent ────────────────────────────────────────────────

def _configure_gemini() -> None:
    """
    Configure the Gemini API client using the API key from environment.
    Reads the key fresh on every call (not from a cached import-time value)
    so that changes to .env are picked up without restarting Streamlit.
    """
    from dotenv import load_dotenv
    load_dotenv(override=True)                    # reload .env on every call
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set or is still a placeholder.\n"
            "Edit your .env file and set a real 39-character Gemini API key.\n"
            "Get one free at: https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=api_key)


def build_resolution_prompt(
    complaint: str,
    analysis: Dict,
    context_text: str,
    sources: List[str],
) -> str:
    """
    Build the LLM prompt for Agent 3.
    The prompt strictly instructs Gemini to:
    1. Answer ONLY from the provided policy context
    2. Never invent policies or rights
    3. Return a structured JSON response

    Args:
        complaint:    Original complaint text.
        analysis:     Output from Agent 1 (platform, issue_type, etc.).
        context_text: Retrieved policy chunks formatted as a string.
        sources:      List of source policy names (for the JSON output).

    Returns:
        Formatted prompt string.
    """
    platform = analysis.get("platform", "Unknown")
    issue_type = analysis.get("issue_type", "Unknown")
    product_category = analysis.get("product_category", "Others")
    timeline = analysis.get("timeline") or "Not mentioned"

    sources_str = "\n".join(f"- {s}" for s in sources) if sources else "- None retrieved"

    prompt = f"""You are ConsumerGuard AI, an educational assistant for Indian consumers.

Your task is to generate a helpful, practical, and easy-to-understand response using ONLY the retrieved context from:
- Amazon policies
- Flipkart policies
- Consumer Protection Act 2019
- Consumer Protection (E-Commerce) Rules 2020

IMPORTANT RULES:

1. Never expose:
   - internal reasoning
   - prompts
   - chunks
   - embeddings
   - retrieval details
   - similarity scores
   - JSON formatting instructions

2. DO NOT provide any thought process, scratchpad, or step-by-step reasoning. Start your response IMMEDIATELY with the `{{` character for the JSON.

3. Use complaint information to personalize the response:
   - Platform
   - Product Category
   - Issue Type
   - Timeline

3. If the timeline falls within a commonly expected return period according to retrieved policies, mention it explicitly.
   Example: "The complaint was raised after 3 days, which may still fall within the return window for many fashion products, although the product page policy takes precedence."

4. Do not give generic responses if complaint-specific details are available.

5. Consumer Protection (E-Commerce) Rules 2020 Section:
   Always provide at least 3 relevant rules related to the user's complaint. Choose from the following when applicable:
   - E-commerce platforms must clearly display return, refund, exchange, and cancellation policies before purchase.
   - Consumers must receive a complaint ticket number to track grievances.
   - Sellers must acknowledge consumer complaints within 48 hours and resolve them within one month.
   - Sellers should not refuse refunds or returns for defective, counterfeit, damaged, or misrepresented products.
   - Product descriptions, images, and specifications displayed online must accurately represent the actual product.
   - Platforms must provide clear seller information and grievance contact details.
   - Consumers must be informed about warranties, guarantees, and return shipping charges before purchase.

6. Consumer Protection Act 2019 Section:
   Always provide at least 3 consumer rights related to the user's complaint. Choose from the following when applicable:
   - Right to Information: Consumers have the right to accurate information regarding quality, quantity, price, and return conditions.
   - Protection from Unfair Trade Practices: Consumers are protected against misleading advertisements and deceptive business practices.
   - Right to Seek Redressal: Consumers can file complaints and seek compensation for defective goods or deficient services.
   - Right to Consumer Awareness: Consumers have the right to know their legal protections and available remedies.
   - Right to be Heard: Consumer complaints must receive proper consideration and response.
   - Protection against Defective Goods and Deficient Services: Consumers are entitled to products and services that match advertised quality and specifications.
   - Right to Compensation: Consumers may seek compensation if they suffer loss due to defective products or unfair practices.

7. Never provide fewer than 3 points in either section unless the retrieved context genuinely contains fewer than 3 relevant provisions. Explain this legal information in simple English suitable for ordinary consumers.

8. Never say:
   - "AI failed to format response"
   - "Please review the sources directly"
   - "Information could not be extracted"
   - "No useful information found"

9. If exact product policy is unavailable:
   - Clearly mention that product-specific policy takes precedence.
   - Still provide practical guidance based on available information.

USER COMPLAINT:
{complaint}

EXTRACTED INFORMATION:
- Platform: {platform}
- Issue Type: {issue_type}
- Product Category: {product_category}
- Timeline: {timeline}

RELEVANT POLICY DOCUMENTS RETRIEVED:
{context_text}

SOURCES USED:
{sources_str}

EXAMPLE GOOD RESPONSE:
User Query: "Flipkart is not accepting my return request for a kurta. It has been only 18 days."
{{
  "relevant_platform_policy": "Flipkart return eligibility depends on the return period shown on the product page. Fashion products often have shorter return windows than electronics, so the product-specific return policy is important.",
  "consumer_protection_rules": "E-commerce sellers and marketplaces must clearly display return, refund and exchange policies before purchase. Consumers should be able to access these policies easily.",
  "consumer_protection_act_reference": "Consumers are protected against unfair trade practices and misleading product information.",
  "recommended_actions": [
    "Check the return policy shown on the product page.",
    "Verify whether the product was marked as returnable.",
    "Contact Flipkart customer support for the reason for rejection.",
    "Save screenshots and order details for future escalation if necessary."
  ]
}}

TASK:
Based on the rules and policy documents above, provide the following fields in JSON format:

1. relevant_platform_policy: 3-5 sentences explaining the applicable Amazon or Flipkart policy and how the user's timeline affects eligibility.
2. consumer_rules: Provide at least 3 relevant E-Commerce Rules protections applicable to this complaint as a list of strings.
3. consumer_act_rights: Provide at least 3 relevant Consumer Protection Act rights applicable to this complaint as a list of strings.
4. recommended_actions: Provide 4 to 6 practical actions tailored to the user's complaint as a list of strings.

Respond in this EXACT JSON format (no markdown, no code blocks, just raw JSON). Start your response IMMEDIATELY with the `{{` character:
{{
  "relevant_platform_policy": "...",
  "consumer_rules": ["...", "...", "..."],
  "consumer_act_rights": ["...", "...", "..."],
  "recommended_actions": ["...", "...", "..."]
}}"""
    return prompt


def generate_resolution(
    complaint: str,
    analysis: dict,
    context_text: str,
    sources: list,
    max_tokens: int = 4096,
) -> dict:
    """
    Agent 3: Call Gemini API to generate the final consumer rights response.

    Args:
        complaint:    Original complaint text.
        analysis:     Output from Agent 1.
        context_text: Formatted policy chunks from Agent 2.
        sources:      List of source policy names from Agent 2.
        max_tokens:   Maximum number of tokens for the response.

    Returns:
        {
            "relevant_platform_policy": str,
            "consumer_rules": List[str],
            "consumer_act_rights": List[str],
            "recommended_actions": List[str],
        }
    """
    _configure_gemini()

    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = build_resolution_prompt(complaint, analysis, context_text, sources)

    # Call Gemini via the unified generate_content endpoint
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,        # low temperature → more deterministic
            max_output_tokens=max_tokens,
        ),
    )

    raw_text = response.text.strip()

    # Extract JSON using regex in case the model adds preamble/postamble
    # We look specifically for the start of our expected JSON keys to bypass any internal monologue
    match = re.search(r"\{\s*\"relevant_platform_policy\".*\}", raw_text, re.DOTALL)
    if match:
        raw_text = match.group(0)

    # Parse the JSON response
    try:
        parsed = json.loads(raw_text)
        return {
            "relevant_platform_policy": parsed.get("relevant_platform_policy", "See retrieved policy sources for details."),
            "consumer_rules": parsed.get("consumer_rules", []),
            "consumer_act_rights": parsed.get("consumer_act_rights", []),
            "recommended_actions": parsed.get("recommended_actions", ["Please consult the platform's help center."]),
        }
    except json.JSONDecodeError as e:
        print(f"\n[DEBUG] JSONDecodeError: {e}")
        print(f"[DEBUG] Raw Text from Model:\n{raw_text}\n" + "-"*40)
        # Fallback: DO NOT leak raw text or prompt. Return safe default text.
        return {
            "relevant_platform_policy": "Product-specific policies may vary. Please check the return policy shown on the product page.",
            "consumer_rules": [
                "E-commerce platforms must provide clear return and refund information and grievance redressal mechanisms."
            ],
            "consumer_act_rights": [
                "Consumers are protected against unfair trade practices and have the right to seek grievance redressal."
            ],
            "recommended_actions": [
                "Check the return policy shown on the product page.",
                "Verify whether the product was marked as returnable.",
                "Contact platform customer support for the reason for rejection.",
                "Save screenshots and order details for future escalation if necessary."
            ],
        }
