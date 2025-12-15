"""Unit tests for Rhythm Agent."""

import pytest
from app.services.mir.schema import StyleGuide, Section, DrumPattern
from app.services.agents.rhythm_agent import invoke_rhythm_agent


@pytest.mark.asyncio
async def test_rhythm_agent_generates_valid_pattern():
    """Test that rhythm agent generates a valid DrumPattern."""
    style_guide = StyleGuide(
        genre="jazz",
        subgenre="ballad",
        harmonic_complexity="medium",
        swing=0.55,
        extensions_allowed=["7", "9"],
        tempo_range=(60, 85)
    )

    section = Section(
        name="verse_A",
        bars=(1, 8),
        key="Dm",
        tempo=72,
        energy="medium"
    )

    # Invoke the rhythm agent
    pattern = await invoke_rhythm_agent(style_guide, section, "drums")

    # Assertions
    assert isinstance(pattern, DrumPattern)
    assert pattern.track == "drums"
    assert pattern.section == "verse_A"
    assert len(pattern.hits) > 0
    assert pattern.swing == style_guide.swing or pattern.swing == 0.55


@pytest.mark.asyncio
async def test_rhythm_agent_uses_valid_instruments():
    """Test that rhythm agent only uses valid drum instruments."""
    style_guide = StyleGuide(
        genre="rock",
        subgenre="ballad",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=[],
        tempo_range=(80, 120)
    )

    section = Section(
        name="chorus_B",
        bars=(9, 16),
        key="E",
        tempo=100,
        energy="high"
    )

    pattern = await invoke_rhythm_agent(style_guide, section)

    # Valid drum instruments from the system prompt
    valid_instruments = {
        "kick", "snare", "hihat_closed", "hihat_open",
        "crash", "ride", "tom_low", "tom_mid", "tom_high", "rim"
    }

    for hit in pattern.hits:
        assert hit.instrument in valid_instruments, \
            f"Invalid drum instrument: {hit.instrument}"


@pytest.mark.asyncio
async def test_rhythm_agent_matches_section_range():
    """Test that drum hits are within the section's bar range."""
    style_guide = StyleGuide(
        genre="pop",
        subgenre="upbeat",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=[],
        tempo_range=(100, 130)
    )

    section = Section(
        name="verse_A",
        bars=(1, 8),
        key="C",
        tempo=120,
        energy="medium"
    )

    pattern = await invoke_rhythm_agent(style_guide, section)

    # Check that all hits are within the section's bar range
    for hit in pattern.hits:
        assert section.bars[0] <= hit.bar <= section.bars[1], \
            f"Drum hit at bar {hit.bar} is outside section range {section.bars}"


@pytest.mark.asyncio
async def test_rhythm_agent_has_kick_on_downbeat():
    """Test that pattern typically has kick drum on beat 1."""
    style_guide = StyleGuide(
        genre="rock",
        subgenre="standard",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=[],
        tempo_range=(100, 140)
    )

    section = Section(
        name="verse_A",
        bars=(1, 4),
        key="G",
        tempo=120,
        energy="medium"
    )

    pattern = await invoke_rhythm_agent(style_guide, section)

    # Look for kick drum on beat 1.0
    kick_hits = [hit for hit in pattern.hits if hit.instrument == "kick"]
    beat_1_kicks = [hit for hit in kick_hits if hit.beat == 1.0]

    # Should have at least one kick on beat 1
    assert len(beat_1_kicks) > 0, "No kick drum on beat 1.0"


@pytest.mark.asyncio
async def test_rhythm_agent_has_snare_backbeat():
    """Test that pattern has snare on backbeat (beats 2 and 4)."""
    style_guide = StyleGuide(
        genre="rock",
        subgenre="standard",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=[],
        tempo_range=(100, 140)
    )

    section = Section(
        name="verse_A",
        bars=(1, 4),
        key="G",
        tempo=120,
        energy="medium"
    )

    pattern = await invoke_rhythm_agent(style_guide, section)

    # Look for snare on beats 2.0 or 4.0 (backbeat)
    snare_hits = [hit for hit in pattern.hits if hit.instrument == "snare"]
    backbeat_snares = [hit for hit in snare_hits if hit.beat in [2.0, 4.0]]

    # Should have at least one snare on backbeat
    assert len(backbeat_snares) > 0, "No snare on backbeat (beats 2 or 4)"


@pytest.mark.asyncio
async def test_rhythm_agent_velocity_matches_energy():
    """Test that velocities match the section's energy level."""
    # Test soft energy
    style_guide_soft = StyleGuide(
        genre="jazz",
        subgenre="ballad",
        harmonic_complexity="medium",
        swing=0.55,
        extensions_allowed=["7", "9"],
        tempo_range=(60, 80)
    )

    section_soft = Section(
        name="intro",
        bars=(1, 4),
        key="Dm",
        tempo=65,
        energy="soft"
    )

    pattern_soft = await invoke_rhythm_agent(style_guide_soft, section_soft)

    # Soft energy should have lower velocities (60-80 range)
    avg_velocity_soft = sum(hit.velocity for hit in pattern_soft.hits) / len(pattern_soft.hits)
    assert 50 <= avg_velocity_soft <= 90, \
        f"Soft energy average velocity {avg_velocity_soft} not in expected range"

    # Test high energy
    style_guide_high = StyleGuide(
        genre="rock",
        subgenre="energetic",
        harmonic_complexity="simple",
        swing=0.0,
        extensions_allowed=[],
        tempo_range=(120, 160)
    )

    section_high = Section(
        name="chorus_B",
        bars=(17, 24),
        key="E",
        tempo=140,
        energy="climax"
    )

    pattern_high = await invoke_rhythm_agent(style_guide_high, section_high)

    # High energy should have higher velocities (90-110 range)
    avg_velocity_high = sum(hit.velocity for hit in pattern_high.hits) / len(pattern_high.hits)
    assert 80 <= avg_velocity_high <= 120, \
        f"High energy average velocity {avg_velocity_high} not in expected range"

    # High energy should be louder than soft energy
    assert avg_velocity_high > avg_velocity_soft, \
        "High energy should have higher average velocity than soft energy"


@pytest.mark.asyncio
async def test_rhythm_agent_swing_amount():
    """Test that swing amount matches style guide."""
    style_guide_swing = StyleGuide(
        genre="jazz",
        subgenre="bebop",
        harmonic_complexity="complex",
        swing=0.67,
        extensions_allowed=["7", "9", "11", "13"],
        tempo_range=(140, 200)
    )

    section = Section(
        name="verse_A",
        bars=(1, 8),
        key="F",
        tempo=160,
        energy="high"
    )

    pattern = await invoke_rhythm_agent(style_guide_swing, section)

    # Pattern should have swing matching style guide
    assert pattern.swing == 0.67 or abs(pattern.swing - 0.67) < 0.1, \
        f"Pattern swing {pattern.swing} doesn't match style guide swing 0.67"
