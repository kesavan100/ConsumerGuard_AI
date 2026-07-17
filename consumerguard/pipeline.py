"""
ConsumerGuard AI — Pipeline Orchestrator
Wires the three agents together into a single analyze() function.

Flow:
  User complaint
      │
      ▼
  Agent 1 (Complaint Analysis)  ← regex/keywords, no LLM
      │ {platform, issue_type, product_category, timeline}
      ▼
  Agent 2 (Retrieval)           ← all-MiniLM-L6-v2 + ChromaDB
      │ {context_text, sources, best_similarity, chunk_count}
      ▼
  Confidence scoring            ← rule-based (not LLM)
      │ HIGH / MEDIUM / LOW
      ▼
  Agent 3 (Resolution)          ← Gemini API
      │ {relevant_policy, consumer_rights, recommended_action}
      ▼
  Final response dict           ← returned to app.py
"""

from typing import Dict

from consumerguard.agents import analyze_complaint, compute_confidence, generate_resolution
from consumerguard.retriever import retrieve


def analyze(complaint_text: str) -> Dict:
    """
    Main entry point for the ConsumerGuard AI pipeline.
    Orchestrates all three agents and returns a structured response dict.

    Args:
        complaint_text: Raw complaint text entered by the user.

    Returns:
        {
            "platform": str,
            "issue_type": str,
            "product_category": str,
            "timeline": str | None,
            "relevant_policy": str,
            "consumer_rights": str,
            "recommended_action": str,
            "confidence_level": str,      # "HIGH", "MEDIUM", or "LOW"
            "sources": List[str],         # e.g. ["Amazon Return Policy", "CP Act 2019"]
        }

    Raises:
        ValueError: If GEMINI_API_KEY is not configured.
        Exception:  If VectorStore is empty (run ingest first).
    """

    # ── Step 1: Agent 1 — Complaint Analysis ──────────────────────────────
    # Rule-based extraction — no LLM, no API cost
    print(f"\n[Pipeline] Step 1: Complaint Analysis...")
    analysis = analyze_complaint(complaint_text)
    print(f"[Pipeline] -> Platform: {analysis['platform']}")
    print(f"[Pipeline] -> Issue Type: {analysis['issue_type']}")
    print(f"[Pipeline] -> Category: {analysis['product_category']}")
    print(f"[Pipeline] -> Timeline: {analysis['timeline']}")

    # ── Step 2: Agent 2 — Retrieval ───────────────────────────────────────
    # Use complaint text directly as the query for simplicity.
    # The platform hint is used to filter ChromaDB results.
    print(f"\n[Pipeline] Step 2: Retrieval...")
    platform_hint = analysis["platform"].lower()  # "amazon", "flipkart", or "unknown"
    retrieval_result = retrieve(
        query=complaint_text,
        platform_hint=platform_hint,
    )
    print(f"[Pipeline] -> Chunks retrieved: {retrieval_result['chunk_count']}")
    print(f"[Pipeline] -> Best similarity: {retrieval_result['best_similarity']}")
    print(f"[Pipeline] -> Sources: {retrieval_result['sources']}")

    # ── Step 3: Confidence Scoring (Rule-Based) ───────────────────────────
    # Computed BEFORE calling Gemini — not invented by the LLM
    confidence_level, confidence_reason = compute_confidence(
        platform=analysis["platform"],
        best_similarity=retrieval_result["best_similarity"],
        chunk_count=retrieval_result["chunk_count"],
    )
    print(f"\n[Pipeline] Step 3: Confidence = {confidence_level}")

    # ── Step 4: Agent 3 — Resolution ─────────────────────────────────────
    # Gemini generates the final response grounded in retrieved context
    if retrieval_result["chunk_count"] == 0:
        print("\n[Pipeline] Step 4: Skipping Gemini (No relevant context retrieved)")
        resolution = {
            "relevant_platform_policy": "No relevant policy information was found in the current knowledge base.",
            "consumer_rules": ["Unsupported query."],
            "consumer_act_rights": ["Unsupported query."],
            "recommended_actions": ["Please check the product page or contact customer support for product-specific policies."]
        }
    else:
        # Dynamic Context & Token scaling
        if confidence_level == "HIGH":
            print("\n[Pipeline] Step 4: Calling Gemini (FAST PATH: top 2 chunks, max_tokens=250)...")
            
            # Truncate context to top 2 chunks only
            top_2_chunks = retrieval_result["chunks"][:2]
            context_parts = []
            for i, chunk in enumerate(top_2_chunks, start=1):
                context_parts.append(
                    f"[Source {i}: {chunk['policy_name']} (similarity: {chunk['similarity_score']})]\n{chunk['text']}"
                )
            context_text = "\n\n---\n\n".join(context_parts)
            
            resolution = generate_resolution(
                complaint=complaint_text,
                analysis=analysis,
                context_text=context_text,
                sources=retrieval_result["sources"],
                max_tokens=250,
            )
        else:
            print(f"\n[Pipeline] Step 4: Calling Gemini ({__import__('consumerguard.config', fromlist=['GEMINI_MODEL']).GEMINI_MODEL})...")
            resolution = generate_resolution(
                complaint=complaint_text,
                analysis=analysis,
                context_text=retrieval_result["context_text"],
                sources=retrieval_result["sources"],
            )
        print(f"[Pipeline] -> Resolution generated successfully.")

    # ── Step 5: Assemble Final Response ──────────────────────────────────
    return {
        "platform": analysis["platform"],
        "issue_type": analysis["issue_type"],
        "product_category": analysis["product_category"],
        "timeline": analysis["timeline"],
        "relevant_platform_policy": resolution["relevant_platform_policy"],
        "consumer_rules": resolution.get("consumer_rules", []),
        "consumer_act_rights": resolution.get("consumer_act_rights", []),
        "recommended_actions": resolution["recommended_actions"],
        "confidence_level": confidence_level,
        "confidence_reason": confidence_reason,
        "matched_documents": retrieval_result["sources"],
    }


# ─── Quick Smoke Test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run a quick end-to-end test with sample complaints.
    Usage: python -m consumerguard.pipeline
    Make sure you have:
    1. Run `python -m consumerguard.ingest` first to build chroma_db/
    2. Set GEMINI_API_KEY in your .env file
    """
    import json
    from dotenv import load_dotenv
    load_dotenv()

    test_complaints = [
        # Test 1: Amazon defective electronics
        "Amazon refused replacement for my laptop after 3 days.",

        # Test 2: Flipkart return issue
        "Flipkart says my kurta cannot be returned after 10 days.",

        # Test 3: Pickup issue
        "Amazon pickup agent refused to collect my damaged monitor.",
        
        # Test 4: Unrelated query
        "My dog is sick.",
        
        # Test 5: Unrelated query 2
        "How do I bake a cake?"
    ]

    for i, complaint in enumerate(test_complaints, start=1):
        print("\n" + "=" * 70)
        print(f"TEST {i}: {complaint[:80]}...")
        print("=" * 70)
        try:
            result = analyze(complaint)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"ERROR: {e}")
