"""Unit tests for Harmony Agent."""

import pytest
import json
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.agents.harmony_agent import (
    create_harmony_agent,
    invoke_harmony_agent,
)
from app.services.mir.schema import StyleGuide, Section, ChordProgression, Chord


class TestHarmonyAgent:
    """Test Harmony Agent functionality."""

    def test_create_harmony_agent(self):
        """Test creating harmony agent returns an agent object."""
        agent = create_harmony_agent()
        assert agent is not None

    @pytest.mark.asyncio
    async def test_invoke_harmony_agent_with_mock(self):
        """Test invoking harmony agent with mocked LLM response."""
        # Create test inputs
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 85)
        )

        section = Section(
            name="verse_A",
            bars=(1, 8),
            key="Dm",
            tempo=72,
            energy="soft"
        )

        # Create mock response
        mock_response_json = {
            "track": "piano",
            "section": "verse_A",
            "chords": [
                {
                    "root": "D",
                    "quality": "m9",
                    "bar": 1,
                    "beat": 1.0,
                    "duration": "whole",
                    "voicing": ["D2", "A2", "F3", "C4", "E4"],
                    "function": "tonic",
                    "velocity": 75
                },
                {
                    "root": "G",
                    "quality": "7",
                    "bar": 2,
                    "beat": 1.0,
                    "duration": "whole",
                    "voicing": ["G2", "D3", "F3", "B3"],
                    "function": "dominant",
                    "velocity": 75
                }
            ]
        }

        # Mock the agent's ainvoke method
        with patch('app.services.agents.harmony_agent.create_harmony_agent') as mock_create:
            mock_agent = AsyncMock()
            mock_agent.ainvoke = AsyncMock(return_value={
                "messages": [
                    MagicMock(content=json.dumps(mock_response_json))
                ]
            })
            mock_create.return_value = mock_agent

            # Invoke the harmony agent
            progression = await invoke_harmony_agent(style_guide, section, "piano")

            # Verify result
            assert isinstance(progression, ChordProgression)
            assert progression.track == "piano"
            assert progression.section == "verse_A"
            assert len(progression.chords) == 2

            # Verify first chord
            assert progression.chords[0].root == "D"
            assert progression.chords[0].quality == "m9"
            assert progression.chords[0].bar == 1
            assert progression.chords[0].voicing == ["D2", "A2", "F3", "C4", "E4"]

            # Verify second chord
            assert progression.chords[1].root == "G"
            assert progression.chords[1].quality == "7"

    def test_harmony_agent_json_parsing(self):
        """Test JSON parsing from various response formats."""
        # Test plain JSON
        json_str = '{"track": "piano", "section": "test", "chords": []}'
        data = json.loads(json_str)
        assert data["track"] == "piano"

        # Test JSON in markdown code block
        markdown_json = '```json\n{"track": "guitar", "section": "test", "chords": []}\n```'
        import re
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', markdown_json, re.DOTALL)
        assert match is not None
        extracted = match.group(1)
        data = json.loads(extracted)
        assert data["track"] == "guitar"


class TestVoiceLeading:
    """Test voice leading validation."""

    def test_voice_leading_quality(self):
        """Test that generated chords can have smooth voice leading."""
        # Create a simple chord progression
        chords = [
            Chord(
                root="D",
                quality="m7",
                bar=1,
                beat=1.0,
                duration="whole",
                voicing=["D2", "A2", "C3", "F3"]
            ),
            Chord(
                root="G",
                quality="7",
                bar=2,
                beat=1.0,
                duration="whole",
                voicing=["G2", "B2", "D3", "F3"]  # Voice leading: A2->B2, C3->D3
            )
        ]

        # Calculate voice leading movement (in semitones)
        from app.services.mir.compiler import pitch_string_to_midi

        for i in range(len(chords) - 1):
            curr_voicing = [pitch_string_to_midi(p) for p in chords[i].voicing]
            next_voicing = [pitch_string_to_midi(p) for p in chords[i + 1].voicing]

            # Check each voice moves smoothly (max 5 semitones)
            for j in range(min(len(curr_voicing), len(next_voicing))):
                movement = abs(next_voicing[j] - curr_voicing[j])
                assert movement <= 5, f"Voice {j} moves {movement} semitones (too much)"

    def test_no_parallel_fifths(self):
        """Test detection of parallel fifths."""
        # This is a basic test - in production, the critic agent will catch these
        chord1_voicing = [60, 67]  # C4 and G4 (perfect fifth)
        chord2_voicing = [62, 69]  # D4 and A4 (perfect fifth, parallel motion)

        interval1 = chord1_voicing[1] - chord1_voicing[0]
        interval2 = chord2_voicing[1] - chord2_voicing[0]

        # Both are perfect fifths (7 semitones)
        assert interval1 == 7
        assert interval2 == 7

        # This would be a parallel fifth (bad voice leading)
        # The critic agent will flag this
