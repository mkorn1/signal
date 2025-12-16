"""Unit tests for Arranger Agent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.agents.arranger_agent import invoke_arranger_agent
from app.services.mir.schema import StyleGuide, Section


class TestArrangerAgent:
    """Test Arranger Agent functionality."""

    @pytest.mark.asyncio
    async def test_invoke_arranger_agent_jazz(self):
        """Test invoking arranger agent for jazz composition."""
        style_guide = StyleGuide(
            genre="jazz",
            subgenre="ballad",
            harmonic_complexity="complex",
            swing=0.55,
            extensions_allowed=["7", "9", "11", "13"],
            tempo_range=(60, 85)
        )

        mock_response = [
            {
                "name": "intro",
                "bars": [1, 8],
                "key": "Dm",
                "tempo": 72,
                "energy": "soft"
            },
            {
                "name": "verse_A",
                "bars": [9, 24],
                "key": "Dm",
                "tempo": 72,
                "energy": "building"
            },
            {
                "name": "outro",
                "bars": [25, 32],
                "key": "Dm",
                "tempo": 72,
                "energy": "soft"
            }
        ]

        with patch('app.services.agents.arranger_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=str(mock_response).replace("'", '"')
            ))
            mock_chat.return_value = mock_instance

            sections = await invoke_arranger_agent(style_guide, target_length_bars=32)

            assert isinstance(sections, list)
            assert len(sections) == 3
            assert all(isinstance(s, Section) for s in sections)

            # Verify first section
            assert sections[0].name == "intro"
            assert sections[0].bars == (1, 8)
            assert sections[0].key == "Dm"
            assert sections[0].energy == "soft"

            # Verify second section
            assert sections[1].name == "verse_A"
            assert sections[1].bars == (9, 24)
            assert sections[1].energy == "building"

    @pytest.mark.asyncio
    async def test_arranger_non_overlapping_bars(self):
        """Test that sections don't overlap in bar ranges."""
        style_guide = StyleGuide(
            genre="pop",
            subgenre="contemporary",
            harmonic_complexity="medium",
            swing=0.0,
            extensions_allowed=["sus4", "7"],
            tempo_range=(100, 130)
        )

        mock_response = [
            {"name": "intro", "bars": [1, 4], "key": "C", "tempo": 120, "energy": "soft"},
            {"name": "verse", "bars": [5, 12], "key": "C", "tempo": 120, "energy": "medium"},
            {"name": "chorus", "bars": [13, 20], "key": "C", "tempo": 120, "energy": "high"}
        ]

        with patch('app.services.agents.arranger_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=str(mock_response).replace("'", '"')
            ))
            mock_chat.return_value = mock_instance

            sections = await invoke_arranger_agent(style_guide, target_length_bars=20)

            # Verify sections are sequential and non-overlapping
            for i in range(len(sections) - 1):
                curr_end = sections[i].bars[1]
                next_start = sections[i + 1].bars[0]
                # Next section should start after or at current section end
                assert next_start >= curr_end, f"Section {i+1} overlaps with section {i}"

    def test_energy_arc_validation(self):
        """Test that energy arc has variation (not all same)."""
        sections = [
            Section(name="intro", bars=(1, 8), key="Dm", tempo=72, energy="soft"),
            Section(name="verse", bars=(9, 16), key="Dm", tempo=72, energy="medium"),
            Section(name="chorus", bars=(17, 24), key="Dm", tempo=72, energy="high"),
            Section(name="outro", bars=(25, 32), key="Dm", tempo=72, energy="soft")
        ]

        # Get unique energy levels
        energy_levels = set(s.energy for s in sections)

        # Should have at least 2 different energy levels
        assert len(energy_levels) >= 2, "Energy arc should have variation"

        # Typical arc: should have at least one high point
        has_high_energy = any(s.energy in ["high", "climax"] for s in sections)
        assert has_high_energy, "Should have at least one high energy section"

    def test_section_bar_ranges(self):
        """Test section bar ranges are valid."""
        section = Section(
            name="verse_A",
            bars=(9, 24),
            key="Dm",
            tempo=120,
            energy="medium"
        )

        # Start should be less than end
        assert section.bars[0] < section.bars[1]

        # Calculate length
        length = section.bars[1] - section.bars[0] + 1
        assert length == 16  # 24 - 9 + 1 = 16 bars

    @pytest.mark.asyncio
    async def test_arranger_with_markdown(self):
        """Test JSON extraction from markdown code block."""
        style_guide = StyleGuide(
            genre="rock",
            subgenre="classic",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["sus4"],
            tempo_range=(120, 140)
        )

        mock_response = [
            {"name": "intro", "bars": [1, 8], "key": "E", "tempo": 130, "energy": "medium"}
        ]
        json_str = str(mock_response).replace("'", '"')
        markdown_content = f'```json\n{json_str}\n```'

        with patch('app.services.agents.arranger_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=markdown_content
            ))
            mock_chat.return_value = mock_instance

            sections = await invoke_arranger_agent(style_guide, target_length_bars=8)

            assert len(sections) == 1
            assert sections[0].name == "intro"
