"""
Rule-Based Contract Risk Scoring
Accenture Contract Review Challenge

Assigns risk weights to important CUAD clause categories and produces
a contract-level risk score for prioritizing contract review.
"""

# Risk weights
# 3 = high risk
# 2 = medium risk
# 1 = low risk

RISK_WEIGHTS = {
    "Limitation of Liability": 3,
    "Indemnification": 3,
    "IP Ownership Assignment": 3,
    "Uncapped Liability": 3,

    "Termination for Convenience": 2,
    "Change of Control": 2,
    "Auto-Renewal": 2,
    "Non-Compete": 2,
    "Exclusivity": 2,

    "Governing Law": 1,
    "Notice Period": 1,
}


def calculate_risk_score(detected_clauses):
    """
    Calculate a risk score based on detected contract clauses.

    Parameters
    ----------
    detected_clauses : list
        List of clause categories detected in a contract.

    Returns
    -------
    int
        Total contract risk score.
    """

    score = 0

    for clause in detected_clauses:
        score += RISK_WEIGHTS.get(clause, 0)

    return score


def assign_risk_level(score):
    """
    Convert numerical risk score into a risk category.
    """

    if score >= 8:
        return "HIGH"

    elif score >= 4:
        return "MEDIUM"

    else:
        return "LOW"


def analyze_contract(contract_id, detected_clauses):
    """
    Generate a simple risk analysis for one contract.
    """

    score = calculate_risk_score(detected_clauses)
    level = assign_risk_level(score)

    return {
        "contract_id": contract_id,
        "detected_clauses": detected_clauses,
        "risk_score": score,
        "risk_level": level,
    }


if __name__ == "__main__":

    # Example contract for testing
    example_clauses = [
        "Indemnification",
        "Auto-Renewal",
        "Change of Control",
        "Governing Law",
    ]

    result = analyze_contract(
        contract_id="example_contract_001",
        detected_clauses=example_clauses,
    )

    print("Contract Risk Analysis")
    print("----------------------")
    print("Contract:", result["contract_id"])
    print("Detected Clauses:", result["detected_clauses"])
    print("Risk Score:", result["risk_score"])
    print("Risk Level:", result["risk_level"])
