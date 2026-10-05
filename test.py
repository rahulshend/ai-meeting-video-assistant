from dotenv import load_dotenv

# .env load BEFORE any core imports
load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.vector_store import build_vector_store
from core.rag_engine import build_rag_chain, ask_question


# ============================================================
# INPUT
# ============================================================

source = "https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s"
language = "english"   # "english" -> Whisper, "hinglish" -> Sarvam


# ============================================================
# 1. DOWNLOAD + AUDIO PROCESSING
# ============================================================

chunks = process_input(source)


# ============================================================
# 2. TRANSCRIPTION
# ============================================================

transcript = transcribe_all(chunks, language=language)

print("\n" + "=" * 60)
print("📝 TRANSCRIPT")
print("=" * 60)

print(
    transcript[:500] + "..."
    if len(transcript) > 500
    else transcript
)


# ============================================================
# 3. BUILD VECTOR DATABASE
# ============================================================

print("\n" + "=" * 60)
print("🧠 BUILDING VECTOR DATABASE")
print("=" * 60)

vector_store = build_vector_store(transcript)

print("✅ Vector database created successfully!")
print("📁 Location: ./vector_db")


# ============================================================
# 4. RAG CHAIN
# ============================================================

print("\n" + "=" * 60)
print("🔗 BUILDING RAG CHAIN")
print("=" * 60)

rag_chain = build_rag_chain(transcript)

print("✅ RAG chain created successfully!")


# ============================================================
# 5. ASK QUESTION USING RAG
# ============================================================

print("\n" + "=" * 60)
print("❓ RAG QUESTION")
print("=" * 60)

question = "What is this video mainly about?"

try:
    answer = ask_question(rag_chain, question)

    print("\n" + "=" * 60)
    print("🤖 RAG ANSWER")
    print("=" * 60)
    print(answer)

except Exception as e:
    print("\n⚠️ RAG LLM ERROR")
    print(e)


# ============================================================
# 6. TITLE + SUMMARY
# ============================================================

try:
    title = generate_title(transcript)
    summary = summarize(transcript)

    print("\n" + "=" * 60)
    print(f"📌 TITLE: {title}")
    print("=" * 60)

    print("\n📋 SUMMARY")
    print("-" * 60)
    print(summary)

except Exception as e:
    print("\n⚠️ SUMMARY/TITLE ERROR")
    print(e)


# ============================================================
# 7. ACTION ITEMS
# ============================================================

try:
    action_items = extract_action_items(transcript)

    print("\n" + "=" * 60)
    print("✅ ACTION ITEMS")
    print("=" * 60)
    print(action_items)

except Exception as e:
    print("\n⚠️ ACTION ITEMS ERROR")
    print(e)


# ============================================================
# 8. KEY DECISIONS
# ============================================================

try:
    decisions = extract_key_decisions(transcript)

    print("\n" + "=" * 60)
    print("🔑 KEY DECISIONS")
    print("=" * 60)
    print(decisions)

except Exception as e:
    print("\n⚠️ KEY DECISIONS ERROR")
    print(e)


# ============================================================
# 9. OPEN QUESTIONS
# ============================================================

try:
    questions = extract_questions(transcript)

    print("\n" + "=" * 60)
    print("❓ OPEN QUESTIONS")
    print("=" * 60)
    print(questions)

except Exception as e:
    print("\n⚠️ OPEN QUESTIONS ERROR")
    print(e)


print("\n" + "=" * 60)
print("🎉 PIPELINE COMPLETED")
print("=" * 60)