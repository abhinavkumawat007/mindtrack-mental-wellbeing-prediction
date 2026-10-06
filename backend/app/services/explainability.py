"""MindTrack Explainability Engine.

Calculates factor contributions and human-understandable explanations
for any given individual prediction, comparing against population baselines.
"""

from typing import List, Dict, Any
from backend.app.schemas.prediction import ContributingFactor

# Dataset population baselines from 5,000 surveyed students
BASELINE_STATS = {
    'Avg_Daily_Usage_Hours': 5.08,
    'Sleep_Hours_Per_Night': 6.63,
    'Study_Hours': 3.01,
    'Physical_Activity_Hours': 1.75,
    'Daily_Unlocks': 171.5,
    'Age': 20.8
}

def explain_prediction(features: Dict[str, Any], predicted_score: float) -> List[ContributingFactor]:
    """Computes the primary contributing factors that pushed the prediction up or down,

    ensuring actionable, transparent insights for any input profile.
    """
    factors: List[ContributingFactor] = []

    usage = float(features.get('avg_daily_usage_hours', 5.0))
    sleep = float(features.get('sleep_hours_per_night', 7.0))
    stress = str(features.get('stress_level', 'Medium'))
    study = float(features.get('study_hours', 3.0))
    physical = float(features.get('physical_activity_hours', 1.5))
    unlocks = int(features.get('daily_unlocks', 150))

    # 1. Screen Usage Influence
    usage_diff = usage - BASELINE_STATS['Avg_Daily_Usage_Hours']
    if usage_diff > 1.0:
        factors.append(ContributingFactor(
            factor="Elevated Screen Time",
            impact="negative",
            description=f"Daily usage of {usage:.1f}h is {usage_diff:+.1f}h above peer average ({BASELINE_STATS['Avg_Daily_Usage_Hours']}h), dampening the score.",
            weight_score=-round(min(abs(usage_diff) * 0.22, 1.2), 2)
        ))
    elif usage_diff < -0.5:
        factors.append(ContributingFactor(
            factor="Restrained Screen Time",
            impact="positive",
            description=f"Controlled screen use of {usage:.1f}h ({abs(usage_diff):.1f}h below average) protects mental bandwidth.",
            weight_score=round(min(abs(usage_diff) * 0.20, 1.0), 2)
        ))
    else:
        factors.append(ContributingFactor(
            factor="Typical Screen Duration",
            impact="positive" if predicted_score >= 6.5 else "neutral",
            description=f"Daily screen time of {usage:.1f}h aligns closely with the cohort norm ({BASELINE_STATS['Avg_Daily_Usage_Hours']}h).",
            weight_score=0.15 if predicted_score >= 6.5 else 0.0
        ))

    # 2. Sleep Hours Influence
    sleep_diff = sleep - BASELINE_STATS['Sleep_Hours_Per_Night']
    if sleep >= 7.0:
        factors.append(ContributingFactor(
            factor="Restorative Sleep Schedule",
            impact="positive",
            description=f"Sufficient sleep ({sleep:.1f}h/night) acts as a strong buffer against cognitive fatigue and mood dips.",
            weight_score=round(min(max(sleep_diff * 0.25, 0.25), 1.1), 2)
        ))
    elif sleep <= 5.8:
        factors.append(ContributingFactor(
            factor="Sleep Deficit",
            impact="negative",
            description=f"Nightly sleep of {sleep:.1f}h is below recommended recovery thresholds, depressing wellbeing.",
            weight_score=-round(min(abs(sleep_diff) * 0.28, 1.3), 2)
        ))
    else:
        factors.append(ContributingFactor(
            factor="Moderate Sleep Duration",
            impact="positive" if sleep >= 6.5 else "neutral",
            description=f"Sleep duration of {sleep:.1f}h provides baseline physical recovery.",
            weight_score=0.2 if sleep >= 6.5 else 0.0
        ))

    # 3. Stress Level Influence
    if stress == 'Very High':
        factors.append(ContributingFactor(
            factor="Acute Stress Burden",
            impact="negative",
            description="Perceived 'Very High' stress level strongly pulls down overall wellbeing rating.",
            weight_score=-1.15
        ))
    elif stress == 'High':
        factors.append(ContributingFactor(
            factor="Elevated Stress",
            impact="negative",
            description="Sustained high stress exerts negative pressure on emotional equilibrium.",
            weight_score=-0.70
        ))
    elif stress == 'Low':
        factors.append(ContributingFactor(
            factor="Low Perceived Stress",
            impact="positive",
            description="Self-reported low stress environment strongly elevates resilience score.",
            weight_score=0.85
        ))
    else:  # Medium
        factors.append(ContributingFactor(
            factor="Manageable Stress Level",
            impact="positive" if predicted_score >= 6.0 else "neutral",
            description="Self-reported medium stress reflects an active but manageable daily cognitive load.",
            weight_score=0.25 if predicted_score >= 6.0 else -0.1
        ))

    # 4. Physical Activity Influence
    phys_diff = physical - BASELINE_STATS['Physical_Activity_Hours']
    if physical >= 1.5:
        factors.append(ContributingFactor(
            factor="Active Movement Routine",
            impact="positive",
            description=f"Consistent movement ({physical:.1f}h/day) enhances dopamine regulation and physical vitality.",
            weight_score=round(min(max(phys_diff * 0.18, 0.2), 0.7), 2)
        ))
    elif physical < 0.8:
        factors.append(ContributingFactor(
            factor="Sedentary Lifestyle",
            impact="negative",
            description="Low daily physical movement reduces natural mood stabilization.",
            weight_score=-0.35
        ))

    # 5. Daily Unlocks & Screen Fragmentation
    if unlocks > 200:
        factors.append(ContributingFactor(
            factor="Frequent Phone Checking",
            impact="negative",
            description=f"{unlocks} daily unlocks indicate frequent attentional fragmentation and compulsive checking.",
            weight_score=-0.45
        ))
    elif unlocks <= 120:
        factors.append(ContributingFactor(
            factor="Focused Attention (Low Unlocks)",
            impact="positive",
            description=f"Only {unlocks} daily unlocks reflects uninterrupted focus and intentional device interaction.",
            weight_score=0.35
        ))

    # Sort factors by absolute magnitude of weight_score so most influential appear first
    factors = sorted(factors, key=lambda f: abs(f.weight_score), reverse=True)
    return factors[:4]
