import os
import re
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy.orm import Session
from app.models import models
from app.services import vector_service

_curr_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_curr_dir, "..", "..", ".env"), override=True)
load_dotenv(os.path.join(_curr_dir, "..", "..", "..", ".env"), override=True)


class AIService:
    def __init__(self):
        self.client = None
        self._init_client()

    def _init_client(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if self.api_key:
            # Strip extra quotes if any
            clean_key = self.api_key.strip().strip('"').strip("'")
            try:
                self.client = Groq(api_key=clean_key)
            except Exception as e:
                print(f"Error initializing Groq client: {e}")
                self.client = None

    # ------------------------------------------------------------------
    # RAG Context Retrieval — Semantic (ChromaDB) with keyword fallback
    # ------------------------------------------------------------------
    def _retrieve_context(self, query: str, db: Session) -> str:
        """
        1st attempt: semantic similarity search via ChromaDB embeddings.
        Fallback: keyword matching across all SQLite rows.
        Returns a formatted context block injected into the system prompt.
        """
        context_blocks = []

        # --- PRIMARY: ChromaDB semantic search ---
        try:
            hits = vector_service.semantic_search(query, n_results=6)
            if hits:
                # Filter by distance threshold (cosine distance < 0.7 = relevant)
                relevant = [h for h in hits if h["distance"] < 0.7]
                for h in relevant:
                    context_blocks.append(h["text"])
        except Exception as e:
            print(f"[AIService] Semantic search error: {e}")

        # --- FALLBACK: keyword matching if ChromaDB has no results ---
        if not context_blocks and db is not None:
            keywords = [w.lower() for w in re.split(r'\W+', query) if len(w) > 2]
            if keywords:
                try:
                    for cs in db.query(models.CaseStudyItem).all():
                        s = f"{cs.title} {cs.category} {cs.summary} {cs.legal_issue} {cs.verdict} {cs.key_principle}".lower()
                        if any(k in s for k in keywords):
                            context_blocks.append(
                                f"[CASE STUDY] {cs.title} ({cs.year})\n"
                                f"Category: {cs.category} | Court: {cs.court}\n"
                                f"Legal Issue: {cs.legal_issue}\nVerdict: {cs.verdict}\n"
                                f"Key Principle: {cs.key_principle}"
                            )
                    for h in db.query(models.CourtHearing).all():
                        s = f"{h.case_title} {h.court_name} {h.hearing_type} {h.notes or ''}".lower()
                        if any(k in s for k in keywords):
                            context_blocks.append(
                                f"[COURT HEARING] {h.case_title}\n"
                                f"Court: {h.court_name} | Date: {h.hearing_date} | Status: {h.status}"
                            )
                    for d in db.query(models.LegalDraft).all():
                        s = f"{d.title} {d.category} {d.content}".lower()
                        if any(k in s for k in keywords):
                            excerpt = (d.content[:400] + "...") if len(d.content) > 400 else d.content
                            context_blocks.append(f"[LEGAL DRAFT] {d.title}\nCategory: {d.category}\nContent: {excerpt}")
                    for ev in db.query(models.EvidenceItem).all():
                        s = f"{ev.case_title} {ev.title} {ev.description}".lower()
                        if any(k in s for k in keywords):
                            context_blocks.append(
                                f"[EVIDENCE] {ev.title} (Case: {ev.case_title})\n"
                                f"Type: {ev.evidence_type} | Status: {ev.status}\nDescription: {ev.description}"
                            )
                except Exception:
                    pass

        if not context_blocks:
            return ""

        joined = "\n\n".join(context_blocks[:6])
        return (
            "=== RETRIEVED CONTEXT FROM DATABASE (Semantic RAG) ===\n"
            f"{joined}\n"
            "=== END OF RETRIEVED CONTEXT ===\n"
        )

    # ------------------------------------------------------------------
    # Main Response Generation (RAG-aware)
    # ------------------------------------------------------------------
    def generate_response(self, user_message: str, history: list, db: Session = None) -> str:
        if not self.client:
            return (
                "I don't have information for that. "
                "(Note: API key is missing or invalid. Please configure the GROQ_API_KEY in the backend.)"
            )

        # 1. Retrieve relevant context from the database
        rag_context = ""
        if db is not None:
            rag_context = self._retrieve_context(user_message, db)

        # 2. Build professional Senior Legal Counsel system prompt
        base_lawyer_instructions = (
            "You are a Senior Legal Counsel and seasoned Advocate serving as the user's dedicated AI Legal Advisor for Nepal Law.\n"
            "Your tone is natural, professional, clear, and direct.\n\n"
            "GUIDELINES:\n"
            "1. **Natural & Direct Answers**: For general questions, explanations, greetings, or conversational questions (e.g. 'what is bail?', 'what happens if I am arrested?', 'namaste'), answer in natural, clear paragraphs and concise bullet points. Do NOT force boilerplate headers or rigid 4-part templates for simple questions.\n"
            "2. **Strict Legal Inquiries**: For complex cases, formal legal assessments, or contract reviews, structure your legal opinion with applicable statutes (e.g., Muluki Criminal/Civil Code, Evidence Act) and actionable next steps.\n"
            "3. **Document & Evidence Analysis**: When analyzing uploaded documents or evidence, directly analyze key terms, obligations, liabilities, and enforceability.\n"
            "4. **No Clutter**: Never output raw markdown table borders (|---|), HTML tags (<br>), or reasoning tags (<think>). Use clean bullet points and bold text."
        )

        if rag_context:
            system_content = (
                f"{base_lawyer_instructions}\n\n"
                "RELEVANT NEPAL CASE PRECEDENTS & STATUTES:\n"
                f"{rag_context}\n"
                "Integrate these retrieved facts and precedents naturally into your answer."
            )
        else:
            system_content = base_lawyer_instructions

        system_prompt = {"role": "system", "content": system_content}

        # 3. Compose message payload
        messages = [system_prompt]
        for msg in history:
            role = "assistant" if msg["role"] == "ai" else msg["role"]
            messages.append({"role": role, "content": msg["content"]})

        if not history or history[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        # 4. Try models in fallback order
        models_to_try = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]
        last_error = ""
        for model_name in models_to_try:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    temperature=0.3,
                    max_tokens=1500,
                )
                raw_content = chat_completion.choices[0].message.content
                clean_content = re.sub(r'<think>.*?(?:</think>|$)', '', raw_content, flags=re.DOTALL).strip()
                if clean_content:
                    return clean_content
            except Exception as e:
                last_error = str(e)
                print(f"Model {model_name} failed: {e}. Trying fallback...")
                continue

        return f"Error communicating with AI service: {last_error}"
