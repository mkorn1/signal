"""Unit tests for Style Agent."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.agents.style_agent import invoke_style_agent
from app.services.mir.schema import StyleGuide


class TestStyleAgent:
    """Test Style Agent functionality."""

    @pytest.mark.asyncio
    async def test_invoke_style_agent_jazz(self):
        """Test invoking style agent with jazz description."""
        mock_response_json = {
            "genre": "jazz",
            "subgenre": "ballad",
            "harmonic_complexity": "complex",
            "swing": 0.55,
            "extensions_allowed": ["7", "9", "11", "13", "b9", "#11"],
            "tempo_range": [60, 85],
            "reference_artists": ["Bill Evans", "Chet Baker"]
        }

        with patch('app.services.agents.style_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=f'{mock_response_json}'.replace("'", '"')
            ))
            mock_chat.return_value = mock_instance

            style_guide = await invoke_style_agent("jazz ballad")

            assert isinstance(style_guide, StyleGuide)
            assert style_guide.genre == "jazz"
            assert style_guide.subgenre == "ballad"
            assert style_guide.harmonic_complexity == "complex"
            assert style_guide.swing == 0.55
            assert "9" in style_guide.extensions_allowed
            assert style_guide.tempo_range == (60, 85)

    @pytest.mark.asyncio
    async def test_invoke_style_agent_pop(self):
        """Test invoking style agent with pop description."""
        mock_response_json = {
            "genre": "pop",
            "subgenre": "contemporary",
            "harmonic_complexity": "medium",
            "swing": 0.0,
            "extensions_allowed": ["sus4", "add9", "7"],
            "tempo_range": [100, 130],
            "reference_artists": []
        }

        with patch('app.services.agents.style_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=f'{mock_response_json}'.replace("'", '"')
            ))
            mock_chat.return_value = mock_instance

            style_guide = await invoke_style_agent("upbeat pop")

            assert isinstance(style_guide, StyleGuide)
            assert style_guide.genre == "pop"
            assert style_guide.harmonic_complexity == "medium"
            assert style_guide.swing == 0.0  # Pop is straight feel
            assert "sus4" in style_guide.extensions_allowed

    @pytest.mark.asyncio
    async def test_invoke_style_agent_with_markdown(self):
        """Test JSON extraction from markdown code block."""
        mock_response_json = {
            "genre": "rock",
            "subgenre": "classic",
            "harmonic_complexity": "simple",
            "swing": 0.0,
            "extensions_allowed": ["sus4"],
            "tempo_range": [120, 140],
            "reference_artists": ["Led Zeppelin"]
        }

        json_str = str(mock_response_json).replace("'", '"')
        markdown_content = f'```json\n{json_str}\n```'

        with patch('app.services.agents.style_agent.ChatOpenAI') as mock_chat:
            mock_instance = AsyncMock()
            mock_instance.ainvoke = AsyncMock(return_value=MagicMock(
                content=markdown_content
            ))
            mock_chat.return_value = mock_instance

            style_guide = await invoke_style_agent("classic rock")

            assert isinstance(style_guide, StyleGuide)
            assert style_guide.genre == "rock"
            assert style_guide.harmonic_complexity == "simple"

    def test_genre_characteristics(self):
        """Test that different genres have appropriate characteristics."""
        # Jazz should have complex harmony
        jazz_style = StyleGuide(
            genre="jazz",
            subgenre="bebop",
            harmonic_complexity="complex",
            swing=0.67,
            extensions_allowed=["7", "9", "11", "13", "b9", "#11"],
            tempo_range=(180, 220)
        )
        assert jazz_style.harmonic_complexity == "complex"
        assert jazz_style.swing > 0.5

        # Pop should have medium complexity, straight feel
        pop_style = StyleGuide(
            genre="pop",
            subgenre="contemporary",
            harmonic_complexity="medium",
            swing=0.0,
            extensions_allowed=["sus4", "add9"],
            tempo_range=(100, 130)
        )
        assert pop_style.harmonic_complexity == "medium"
        assert pop_style.swing == 0.0

        # Rock should have simple harmony
        rock_style = StyleGuide(
            genre="rock",
            subgenre="hard",
            harmonic_complexity="simple",
            swing=0.0,
            extensions_allowed=["sus4"],
            tempo_range=(120, 160)
        )
        assert rock_style.harmonic_complexity == "simple"
