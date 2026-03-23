def build_report(
    snapshot: dict,
    news_items: list,
    rag_result: dict = None,
    backtest: dict = None,
    prediction_result: dict = None,
    risk_result: dict = None,
) -> str:
    lines = [
        f"Ticker: {snapshot.get('ticker')}",
        f"Company: {snapshot.get('company_name')}",
        f"Sector: {snapshot.get('sector')}",
        f"Industry: {snapshot.get('industry')}",
        f"Current Price: {snapshot.get('current_price')}",
        f"Market Cap: {snapshot.get('market_cap')}",
        "",
        "Business Summary:",
        snapshot.get("business_summary") or "No business summary available.",
    ]

    if news_items:
        lines.extend(["", "Recent News:"])
        for i, item in enumerate(news_items, 1):
            title = item.get("title") or (item.get("summary") or "Untitled article")[:100]
            publisher = item.get("publisher") or "Unknown publisher"
            lines.append(f"{i}. {title} | {publisher}")

    if rag_result:
        lines.extend([
            "",
            "RAG Research Answer:",
            rag_result.get("answer", "")
        ])

    if prediction_result:
        p = prediction_result.get("metrics", {})
        lines.extend([
            "",
            "Predictive Analytics:",
            f"Model: {p.get('model')}",
            f"Test Accuracy: {p.get('test_accuracy')}",
            f"Latest Probability Up: {p.get('latest_probability_up')}",
            f"Latest Signal: {p.get('latest_signal')}",
        ])

    if risk_result:
        lines.extend([
            "",
            "Risk Policy Decision:",
            f"Action: {risk_result.get('action')}",
            f"Shares: {risk_result.get('shares')}",
            f"Notional: {risk_result.get('notional')}",
            f"Human Approval Required: {risk_result.get('human_approval_required')}",
            "Reasons:",
        ])
        for r in risk_result.get("reasons", []):
            lines.append(f"- {r}")

    if backtest:
        m = backtest.get("metrics", {})
        lines.extend([
            "",
            "Backtest Metrics:",
            f"Strategy Return: {m.get('strategy_return_pct')}",
            f"Buy and Hold Return: {m.get('buy_and_hold_return_pct')}",
            f"Max Drawdown: {m.get('max_drawdown_pct')}",
        ])

    return "\n".join(lines)