"""
agents.py — Agentic Workflow for GreenGov AI
Implements ProfileAgent, RetrievalAgent, EligibilityAgent, ResponseAgent.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional

import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from rag import retrieve_relevant_chunks, format_context


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class UserProfile:
    """Structured user profile extracted by ProfileAgent."""
    raw_query: str
    state: Optional[str] = None
    land_size: Optional[str] = None
    crop_type: Optional[str] = None
    water_source: Optional[str] = None
    goal: Optional[str] = None
    farmer_category: Optional[str] = None   # small / marginal / large
    is_farmer: bool = True
    extra_context: str = ""


@dataclass
class AgentContext:
    """Shared context object passed through the agent pipeline."""
    user_query: str
    profile: Optional[UserProfile] = None
    retrieved_docs: list = field(default_factory=list)
    context_text: str = ""
    sources: List[str] = field(default_factory=list)
    eligibility_notes: str = ""
    final_response: str = ""


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

PROFILE_EXTRACTION_PROMPT = """
You are a government scheme assistant helping Indian citizens find subsidies.
Extract the following details from the user's query (if mentioned). 
Return ONLY a valid JSON object with these keys — leave missing fields as null:

{{
  "state": "<Indian state name or null>",
  "land_size": "<land size in acres/hectares or null>",
  "crop_type": "<crop or farming type or null>",
  "water_source": "<irrigation/water source or null>",
  "goal": "<main goal: solar/water/seeds/insurance/irrigation/soil or null>",
  "farmer_category": "<small/marginal/large or null>",
  "is_farmer": true or false
}}

User Query: {query}
"""

ELIGIBILITY_PROMPT = """
You are an expert on Indian government agricultural and sustainability schemes.
Based on the user profile and the scheme documents below, analyze eligibility.

User Profile:
- State: {state}
- Land Size: {land_size}
- Crop Type: {crop_type}
- Water Source: {water_source}
- Goal: {goal}
- Farmer Category: {farmer_category}

Scheme Document Excerpts:
{context}

Analyze which schemes the user likely qualifies for and note any eligibility criteria 
they should verify (land holding limits, state-specific applicability, etc.).
Be concise — 3 to 5 bullet points.
"""

RESPONSE_GENERATION_PROMPT = """
You are GreenGov AI — a helpful assistant that helps Indian citizens discover 
government subsidies for clean energy, water conservation, and sustainable agriculture.

User's Question: {query}

User Profile Summary: {profile_summary}

Eligibility Analysis: {eligibility_notes}

Relevant Scheme Information:
{context}

Generate a comprehensive, citizen-friendly response in this exact structure:

## 🌱 Recommended Schemes
List the most relevant schemes by name with a one-line description of each.

## ✅ Eligibility Analysis
Summarize who qualifies and any conditions relevant to this user.

## 💰 Key Benefits
Bullet-point the main financial/material benefits available.

## 📄 Required Documents
List the typical documents needed to apply.

## 📝 Application Steps
Step-by-step instructions (numbered) on how to apply.

## 📚 Sources Used
{sources}

---
⚠️ *This recommendation is based on available scheme documents and should be 
verified through official government portals (pmkisan.gov.in, india.gov.in, etc.).*
"""


# ─────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────

def profile_agent(query: str, gemini_model) -> UserProfile:
    """
    ProfileAgent: Extract structured user profile from the free-text query.
    Uses Gemini to parse intent, state, land size, crop type, and goals.
    """
    import json

    prompt = PROFILE_EXTRACTION_PROMPT.format(query=query)
    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?", "", raw).strip()
        raw = re.sub(r"```$", "", raw).strip()

        data = json.loads(raw)
        profile = UserProfile(
            raw_query=query,
            state=data.get("state"),
            land_size=data.get("land_size"),
            crop_type=data.get("crop_type"),
            water_source=data.get("water_source"),
            goal=data.get("goal"),
            farmer_category=data.get("farmer_category"),
            is_farmer=data.get("is_farmer", True),
        )
    except Exception:
        # Graceful fallback — continue with empty profile
        profile = UserProfile(raw_query=query)

    return profile


def retrieval_agent(query: str, profile: UserProfile, vectorstore: FAISS) -> tuple:
    """
    RetrievalAgent: Build an enriched query from the user profile and retrieve
    the most relevant scheme document chunks from FAISS.
    Returns (docs, context_text, sources).
    """
    # Enrich the query with profile signals for better retrieval
    enriched_parts = [query]
    if profile.goal:
        enriched_parts.append(profile.goal)
    if profile.state:
        enriched_parts.append(profile.state)
    if profile.crop_type:
        enriched_parts.append(profile.crop_type)

    enriched_query = " ".join(enriched_parts)
    docs = retrieve_relevant_chunks(enriched_query, vectorstore, k=6)
    context_text, sources = format_context(docs)
    return docs, context_text, sources


def eligibility_agent(
    profile: UserProfile,
    context_text: str,
    gemini_model,
) -> str:
    """
    EligibilityAgent: Using the retrieved context and user profile, produce
    a brief eligibility analysis to feed into the final response.
    """
    prompt = ELIGIBILITY_PROMPT.format(
        state=profile.state or "Not specified",
        land_size=profile.land_size or "Not specified",
        crop_type=profile.crop_type or "Not specified",
        water_source=profile.water_source or "Not specified",
        goal=profile.goal or "General inquiry",
        farmer_category=profile.farmer_category or "Not specified",
        context=context_text[:3000],   # Keep prompt size manageable
    )
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Eligibility analysis unavailable: {e}"


def response_agent(
    query: str,
    profile: UserProfile,
    eligibility_notes: str,
    context_text: str,
    sources: List[str],
    gemini_model,
) -> str:
    """
    ResponseAgent: Generate the final structured citizen-facing response
    combining all upstream agent outputs.
    """
    profile_summary = (
        f"State: {profile.state or 'N/A'} | "
        f"Land: {profile.land_size or 'N/A'} | "
        f"Crop: {profile.crop_type or 'N/A'} | "
        f"Goal: {profile.goal or 'General'} | "
        f"Category: {profile.farmer_category or 'N/A'}"
    )

    sources_text = (
        "\n".join(f"- {s}" for s in sources)
        if sources else "- General scheme knowledge base"
    )

    prompt = RESPONSE_GENERATION_PROMPT.format(
        query=query,
        profile_summary=profile_summary,
        eligibility_notes=eligibility_notes,
        context=context_text,
        sources=sources_text,
    )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Could not generate response: {e}"


# ─────────────────────────────────────────────
# Orchestrator — Full Agentic Pipeline
# ─────────────────────────────────────────────

def run_agentic_pipeline(
    query: str,
    vectorstore: FAISS,
    gemini_model,
) -> dict:
    """
    Orchestrates the full multi-agent workflow:
    User Query → ProfileAgent → RetrievalAgent → EligibilityAgent → ResponseAgent

    Returns a dict with all intermediate results and the final answer.
    """
    ctx = AgentContext(user_query=query)

    # ── Step 1: Profile Extraction ──────────────
    ctx.profile = profile_agent(query, gemini_model)

    # ── Step 2: Retrieval ───────────────────────
    ctx.retrieved_docs, ctx.context_text, ctx.sources = retrieval_agent(
        query, ctx.profile, vectorstore
    )

    # ── Step 3: Eligibility Analysis ────────────
    ctx.eligibility_notes = eligibility_agent(
        ctx.profile, ctx.context_text, gemini_model
    )

    # ── Step 4: Response Generation ─────────────
    ctx.final_response = response_agent(
        query=query,
        profile=ctx.profile,
        eligibility_notes=ctx.eligibility_notes,
        context_text=ctx.context_text,
        sources=ctx.sources,
        gemini_model=gemini_model,
    )

    return {
        "response": ctx.final_response,
        "profile": ctx.profile,
        "sources": ctx.sources,
        "eligibility": ctx.eligibility_notes,
        "doc_count": len(ctx.retrieved_docs),
    }
