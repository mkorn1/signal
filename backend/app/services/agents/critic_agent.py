"""Critic Agent - Quality control and validation for musical compositions.

The Critic Agent evaluates generated musical content for:
- Technical correctness (voice leading, range, playability)
- Musical quality (motif development, phrasing, groove)
- Style consistency with the StyleGuide

It has authority to send work back for revision if quality thresholds are not met.
"""

from app.services.mir.schema import ChordProgression, MelodyPhrase, DrumPattern, StyleGuide
from app.services.mir.validators import (
    validate_voice_leading,
    validate_melody_range,
    validate_style_consistency,
    validate_all
)
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CriticReport:
    """Critic evaluation report."""
    overall_score: float  # 0.0-1.0
    issues: List[Dict]  # List of errors/warnings with details
    passed: bool  # True if score >= threshold and no errors
    revision_needed: List[str]  # Which agents to re-run: ["harmony", "melody", "rhythm"]
    summary: str  # Human-readable summary of the critique


# Scoring weights
ERROR_PENALTY = 0.2  # Deduct 0.2 from score per error
WARNING_PENALTY = 0.05  # Deduct 0.05 from score per warning
PASS_THRESHOLD = 0.8  # Must score >= 0.8 to pass


async def invoke_critic_agent(
    style_guide: StyleGuide,
    harmony: ChordProgression,
    melody: Optional[MelodyPhrase] = None,
    rhythm: Optional[DrumPattern] = None
) -> CriticReport:
    """Evaluate composition and return critique.

    This is a rule-based critic. Future enhancements may add
    LLM-based musical quality evaluation.

    Args:
        style_guide: StyleGuide to validate against
        harmony: ChordProgression to evaluate
        melody: Optional MelodyPhrase to evaluate
        rhythm: Optional DrumPattern to evaluate

    Returns:
        CriticReport with score, issues, and revision recommendations
    """
    all_issues = []

    # Run rule-based validators
    print(f"[CRITIC] Evaluating harmony...")
    all_issues.extend(validate_voice_leading(harmony))
    all_issues.extend(validate_style_consistency(harmony, style_guide))

    if melody:
        print(f"[CRITIC] Evaluating melody...")
        all_issues.extend(validate_melody_range(melody))

    # TODO: Add rhythm validation in future iterations
    # if rhythm:
    #     print(f"[CRITIC] Evaluating rhythm...")
    #     all_issues.extend(validate_rhythm_groove(rhythm))

    # Count errors vs warnings
    error_count = sum(1 for issue in all_issues if issue.get("severity") == "error")
    warning_count = sum(1 for issue in all_issues if issue.get("severity") == "warning")

    # Calculate score
    # Start at 1.0, deduct ERROR_PENALTY per error, WARNING_PENALTY per warning
    score = max(0.0, 1.0 - (error_count * ERROR_PENALTY) - (warning_count * WARNING_PENALTY))

    # Determine if revision needed
    passed = score >= PASS_THRESHOLD and error_count == 0

    # Group issues by responsible agent
    revision_needed = []
    agent_issues = {}

    for issue in all_issues:
        agent = issue.get("agent", "unknown")
        if agent not in agent_issues:
            agent_issues[agent] = []
        agent_issues[agent].append(issue)

    # Determine which agents need to revise
    if not passed:
        revision_needed = list(agent_issues.keys())

    # Build summary
    summary_parts = [
        f"Overall Score: {score:.2f}/1.00",
        f"Issues: {error_count} errors, {warning_count} warnings"
    ]

    if passed:
        summary_parts.append("✓ PASSED - Quality threshold met")
    else:
        summary_parts.append(f"✗ FAILED - Revision needed from: {', '.join(revision_needed)}")

    if all_issues:
        summary_parts.append("\nTop Issues:")
        for issue in all_issues[:3]:  # Show first 3 issues
            summary_parts.append(f"  - [{issue['severity'].upper()}] {issue['message']} ({issue['location']})")

    summary = "\n".join(summary_parts)

    print(f"[CRITIC] {summary}")

    return CriticReport(
        overall_score=score,
        issues=all_issues,
        passed=passed,
        revision_needed=revision_needed,
        summary=summary
    )


def get_issues_for_agent(report: CriticReport, agent_name: str) -> List[Dict]:
    """Extract issues relevant to a specific agent from a critic report.

    Args:
        report: CriticReport to filter
        agent_name: Name of agent to get issues for ("harmony", "melody", "rhythm")

    Returns:
        List of issues for that agent
    """
    return [issue for issue in report.issues if issue.get("agent") == agent_name]


def format_critique_for_agent(report: CriticReport, agent_name: str) -> str:
    """Format critique feedback for an agent to use in revision.

    Args:
        report: CriticReport with issues
        agent_name: Name of agent

    Returns:
        Formatted string with feedback for the agent
    """
    issues = get_issues_for_agent(report, agent_name)

    if not issues:
        return f"No issues found for {agent_name} agent."

    lines = [f"Critique feedback for {agent_name} agent:"]
    lines.append(f"Found {len(issues)} issue(s):\n")

    for i, issue in enumerate(issues, 1):
        lines.append(f"{i}. [{issue['severity'].upper()}] {issue['type']}")
        lines.append(f"   Location: {issue['location']}")
        lines.append(f"   Message: {issue['message']}")
        if 'suggestion' in issue:
            lines.append(f"   Suggestion: {issue['suggestion']}")
        lines.append("")

    return "\n".join(lines)
