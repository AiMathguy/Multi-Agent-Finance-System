from app.core.config import GROQ_API_KEY, GROQ_MODEL

print("KEY EXISTS:", bool(GROQ_API_KEY))
print("MODEL:", GROQ_MODEL)
print("KEY PREFIX:", GROQ_API_KEY[:10] if GROQ_API_KEY else "NONE")