import os

from app.core.parser import extract_ticker, detect_tasks
from app.services.company import get_company_snapshot
from app.services.news import get_news
from app.services.export import save_company_csv, save_news_csv
from app.services.reporting import build_report
from app.rag.ingest import ingest_company_knowledge
from app.rag.qa import generate_rag_answer
from app.quant.backtest import run_backtest
from app.core.config import OUTPUT_DIR
from app.core.guards import PipelineGuard
from app.ml.predictor import run_random_forest_prediction
from app.ml.risk import risk_policy_decision


def run_pipeline(prompt: str):
    ticker = extract_ticker(prompt)
    tasks = detect_tasks(prompt)

    if not ticker:
        return {"status": "error", "message": "No ticker detected in prompt."}

    guard = PipelineGuard(max_steps=8, cost_cap_usd=1.00)
    debug_steps = []

    debug_steps.append(guard.step("parse_prompt", estimated_cost=0.0))

    snapshot = get_company_snapshot(ticker)
    debug_steps.append(guard.step("get_company_snapshot", estimated_cost=0.0))

    news_items = get_news(ticker, limit=5) if (tasks["news"] or tasks["rag"] or tasks["report"]) else []
    if news_items:
        debug_steps.append(guard.step("get_news", estimated_cost=0.0))

    artifacts = {}
    metrics = {}
    rag_result = None
    backtest_result = None
    prediction_result = None
    risk_result = None

    # Ingest RAG docs first
    if tasks["report"] or tasks["news"] or tasks["rag"]:
        ingested_chunks = ingest_company_knowledge(ticker, snapshot, news_items)
        artifacts["rag_chunks_ingested"] = ingested_chunks
        debug_steps.append(guard.step("ingest_rag_documents", estimated_cost=0.0))

    # Export CSVs if requested
    if tasks["csv"]:
        artifacts["company_csv"] = save_company_csv(snapshot)
        news_csv = save_news_csv(ticker, news_items)
        if news_csv:
            artifacts["news_csv"] = news_csv
        debug_steps.append(guard.step("export_csv", estimated_cost=0.0))

    # Run predictive model before RAG so RAG can see prediction/risk outputs
    if tasks["predictive"]:
        prediction_result = run_random_forest_prediction(ticker=ticker, period="2y")
        artifacts.update(prediction_result.get("artifacts", {}))
        metrics["prediction"] = prediction_result.get("metrics", {})
        debug_steps.append(guard.step("run_random_forest_prediction", estimated_cost=0.0))

        risk_result = risk_policy_decision(
            latest_probability_up=prediction_result["metrics"]["latest_probability_up"],
            current_price=snapshot.get("current_price"),
            portfolio_cash=200000.0,
            max_position_pct=0.10,
            min_confidence_to_buy=0.60,
            max_single_trade_amount=20000.0,
        )
        metrics["risk_decision"] = risk_result
        debug_steps.append(guard.step("apply_risk_policy", estimated_cost=0.0))

    # Run backtest if requested
    if tasks["backtest"]:
        backtest_result = run_backtest(ticker)
        metrics.update(backtest_result.get("metrics", {}))
        artifacts.update(backtest_result.get("artifacts", {}))
        debug_steps.append(guard.step("run_backtest", estimated_cost=0.0))

    # Run RAG after prediction/risk so it can use them in context
    if tasks["rag"]:
        rag_result = generate_rag_answer(
            query=prompt,
            ticker=ticker,
            prediction_result=prediction_result,
            risk_result=risk_result,
        )
        debug_steps.append(guard.step("generate_rag_answer", estimated_cost=0.10))

    report_text = build_report(
        snapshot=snapshot,
        news_items=news_items,
        rag_result=rag_result,
        backtest=backtest_result,
        prediction_result=prediction_result,
        risk_result=risk_result,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_path = os.path.join(OUTPUT_DIR, f"{ticker}_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    artifacts["report_txt"] = report_path

    return {
        "status": "success",
        "ticker": ticker,
        "tasks": tasks,
        "snapshot": snapshot,
        "news": news_items,
        "rag": rag_result,
        "metrics": metrics,
        "artifacts": artifacts,
        "report_text": report_text,
        "human_intervention": {
            "required": True,
            "message": "Any buy proposal is advisory only and requires explicit human approval before execution."
        },
        "guardrails": {
            "max_steps": guard.max_steps,
            "steps_used": guard.steps_used,
            "cost_cap_usd": guard.cost_cap_usd,
            "cost_used_estimate": round(guard.cost_used, 4),
            "step_log": debug_steps,
        }
    }