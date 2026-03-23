from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.core.config import GROQ_API_KEY, GROQ_MODEL
from app.rag.retrieve import retrieve_company_context

def generate_rag_answer(query: str, ticker: str, prediction_result=None, risk_result=None):

    docs = retrieve_company_context(query=query, ticker=ticker, k=5)

    if not docs:
        return {
            "answer": "No retrieved context found for this company yet.",
            "sources": []
        }
    
    context_docs = "\n\n---\n\n".join([d.page_content for d in docs])

    extra_context = ""

    if prediction_result:
        p = prediction_result.get("metrics", {})
        extra_context += f"""
    PREDICTIVE MODEL OUTPUT:
    - Probability Up: {p.get('latest_probability_up')}
    - Signal: {p.get('latest_signal')}
    - Test Accuracy: {p.get('test_accuracy')}
    """

    if risk_result:
        extra_context += f"""
    RISK DECISION:
    - Action: {risk_result.get('action')}
    - Shares: {risk_result.get('shares')}
    - Notional: {risk_result.get('notional')}
    - Reasons: {", ".join(risk_result.get('reasons', []))}
    """

    context = context_docs + "\n\n" + extra_context

    prompt = ChatPromptTemplate.from_template("""
You are a financial research assistant.

Use the provided context to produce a practical decision-support summary.

Rules:
- Use only the provided context.
- Separate facts from interpretation.
- If predictive model output is present, include it explicitly.
- If risk decision output is present, include it explicitly.
- Do not claim you cannot analyze prediction or investment size if those values are present in the context.

Structure your answer as:

Facts:
- ...

Themes:
- ...

Prediction:
- ...

Risk Decision:
- ...

Conclusion:
- ...

User question:
{query}

Retrieved context:
{context}
""")

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0
    )

    chain = prompt | llm
    result = chain.invoke({
        "query": query,
        "context": context
    })

    return {
        "answer": result.content,
        "sources": [d.metadata for d in docs]
    }