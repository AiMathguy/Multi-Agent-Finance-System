def risk_policy_decision(
    latest_probability_up: float,
    current_price: float,
    portfolio_cash: float = 200000.0,
    max_position_pct: float = 0.10,
    min_confidence_to_buy: float = 0.60,
    max_single_trade_amount: float = 20000.0,
):
    reasons = []
    action = "NO_BUY"
    shares = 0
    notional = 0.0
    human_approval_required = True

    if current_price is None or current_price <= 0:
        reasons.append("Current price unavailable or invalid.")
        return {
            "action": action,
            "shares": shares,
            "notional": notional,
            "human_approval_required": human_approval_required,
            "reasons": reasons,
        }

    if latest_probability_up < min_confidence_to_buy:
        reasons.append(
            f"Model confidence {latest_probability_up:.2f} is below threshold {min_confidence_to_buy:.2f}."
        )
        return {
            "action": action,
            "shares": shares,
            "notional": notional,
            "human_approval_required": human_approval_required,
            "reasons": reasons,
        }

    raw_max_by_pct = portfolio_cash * max_position_pct
    allowed_notional = min(raw_max_by_pct, max_single_trade_amount)

    shares = int(allowed_notional // current_price)
    notional = round(shares * current_price, 2)

    if shares <= 0:
        reasons.append("Risk cap allows zero shares at current price.")
        return {
            "action": action,
            "shares": 0,
            "notional": 0.0,
            "human_approval_required": human_approval_required,
            "reasons": reasons,
        }

    action = "PROPOSE_BUY"
    reasons.append(f"Confidence passed threshold at {latest_probability_up:.2f}.")
    reasons.append(f"Position capped by risk policy to {max_position_pct:.0%} of portfolio.")
    reasons.append(f"Single-trade amount capped at {max_single_trade_amount:.2f}.")
    reasons.append("Human approval required before any execution.")

    return {
        "action": action,
        "shares": shares,
        "notional": notional,
        "human_approval_required": human_approval_required,
        "reasons": reasons,
    }