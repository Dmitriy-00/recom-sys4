"""
Data models for materials, users, and tropes
Implements the 89+ parameter material taxonomy and multi-dimensional user profile
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# ============================================
# ENUMS
# ============================================

class MaterialType(Enum):
    FILM = "film"
    BOOK = "book"
    GAME = "game"
    SERIES = "series"


class UsageType(Enum):
    STRAIGHT = "straight"
    DECONSTRUCTION = "deconstruction"
    SUBVERSION = "subversion"
    RECONSTRUCTION = "reconstruction"
    META = "meta"


class CognitiveStyle(Enum):
    VISUAL = "visual"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    KINESTHETIC = "kinesthetic"


# ============================================
# MATERIAL MODEL (89+ parameters)
# ============================================

@dataclass
class MaterialMetadata:
    """Basic metadata for materials"""
    title: str
    original_title: Optional[str] = None
    year: Optional[int] = None
    creator: Optional[str] = None
    country: Optional[str] = None
    language: Optional[str] = None
    type: MaterialType = MaterialType.FILM


@dataclass
class PhysicalParams:
    """Physical parameters"""
    runtime: Optional[int] = None  # минуты для фильмов
    pages: Optional[int] = None  # страницы для книг
    episodes: Optional[int] = None  # эпизоды для сериалов
    asl: Optional[float] = None  # Average Shot Length


@dataclass
class GenreTaxonomy:
    """Genre classification"""
    primary_genre: str = "unknown"
    subgenres: List[str] = field(default_factory=list)
    genre_purity: float = 0.5
    genre_blend: Dict[str, float] = field(default_factory=dict)
    narrative_mode: str = "realistic"
    tone: str = "serious"


@dataclass
class VisualCharacteristics:
    """Visual parameters for films"""
    asl: float = 5.0
    asl_variance: float = 1.0
    color_palette: str = "neutral"
    saturation: float = 5.0
    contrast: float = 5.0
    lighting_style: str = "naturalistic"
    visual_density: float = 5.0
    realism_level: float = 5.0
    stylization: float = 5.0


@dataclass
class TextCharacteristics:
    """Text parameters for books"""
    flesch_reading_ease: Optional[float] = None
    avg_sentence_length: Optional[float] = None
    lexical_diversity: Optional[float] = None
    formality: float = 5.0
    abstractness: float = 5.0
    imagery_density: float = 5.0
    dialogue_to_narration_ratio: float = 0.3


@dataclass
class AudioCharacteristics:
    """Audio parameters"""
    soundtrack_type: str = "orchestral"
    music_density: float = 0.5
    sound_design_complexity: float = 5.0
    dialogue_density: float = 5.0


@dataclass
class PacingStructure:
    """Pacing and structure"""
    tempo_overall: float = 5.0
    tempo_variation: float = 5.0
    act_structure: str = "3-act"
    linearity: float = 1.0
    timeline_complexity: int = 1
    flashback_ratio: float = 0.0


@dataclass
class CharacterParameters:
    """Character parameters"""
    protagonist_archetype: str = "hero"
    transformation_magnitude: float = 5.0
    agency: float = 5.0
    moral_alignment_good_evil: float = 0.5  # -1 evil to 1 good
    moral_alignment_lawful_chaotic: float = 0.0  # -1 chaotic to 1 lawful
    character_count_total: int = 1
    character_count_major: int = 1
    psychological_depth: float = 5.0
    moral_ambiguity: float = 0.5


@dataclass
class ThematicParameters:
    """Thematic content"""
    primary_themes: List[Dict[str, Any]] = field(default_factory=list)
    thematic_density: float = 5.0
    philosophical_depth_overall: float = 5.0
    philosophical_depth_epistemological: float = 5.0
    philosophical_depth_metaphysical: float = 5.0
    philosophical_depth_ethical: float = 5.0
    philosophical_depth_existential: float = 5.0
    intellectual_complexity: float = 5.0
    abstraction_level: float = 5.0
    symbolism_density: float = 5.0

    # Worldview
    worldview_optimism_pessimism: float = 0.0  # -1 to 1
    worldview_determinism_free_will: float = 0.0
    worldview_individualism_collectivism: float = 0.0
    worldview_idealism_pragmatism: float = 0.0


@dataclass
class EmotionalParameters:
    """Emotional parameters"""
    emotional_arc_type: str = "rags_to_riches"
    emotional_volatility: float = 5.0
    final_tone: float = 0.0  # -1 tragedy to 1 optimism
    emotional_diversity: float = 5.0
    emotional_intensity_overall: float = 5.0
    emotional_intensity_peak: float = 7.0
    catharsis_potential: float = 5.0
    atmosphere_primary: str = "neutral"


@dataclass
class TropeUsage:
    """How a trope is used in a material"""
    trope_id: str
    trope_name: str
    category: str = "plot_device"
    usage_type: UsageType = UsageType.STRAIGHT

    # Quality metrics
    awareness: float = 5.0  # 1-10
    execution: float = 5.0  # 1-10
    originality: float = 5.0  # 1-10
    centrality: float = 0.5  # 0-1

    # Impact
    cognitive_load: float = 5.0
    transformation_potential: float = 5.0
    meta_commentary: bool = False


@dataclass
class CognitiveOperations:
    """Cognitive operations activated"""
    operations_activated: List[Dict[str, float]] = field(default_factory=list)
    cognitive_style_match_visual: float = 0.5
    cognitive_style_match_analytical: float = 0.5
    cognitive_style_match_empathetic: float = 0.5
    cognitive_style_match_kinesthetic: float = 0.5


@dataclass
class TransformativePotential:
    """Transformative potential metrics"""
    transformative_score: float = 5.0  # 0-10

    # Components
    cognitive_dissonance: float = 0.5
    perspective_shifting: float = 0.5
    metacognitive_awareness: float = 0.5
    multiple_perspectives: float = 0.5

    # Journey
    entry_barrier: float = 5.0
    peak_challenge: float = 7.0
    integration_support: float = 5.0
    lasting_impact_potential: float = 0.5

    # Value
    rewatch_value: float = 5.0
    discussion_potential: float = 5.0


@dataclass
class CulturalContext:
    """Cultural context and impact"""
    historical_period: Optional[str] = None
    cultural_moment: Optional[str] = None
    influenced_by: List[str] = field(default_factory=list)
    part_of_movement: Optional[str] = None
    mainstream_penetration: float = 0.5
    critical_acclaim: float = 5.0
    canonical_status: float = 0.5


@dataclass
class AudienceParameters:
    """Audience and accessibility"""
    age_recommendation: str = "16+"
    education_level: str = "high_school"
    cultural_knowledge_required: List[str] = field(default_factory=list)
    attention_required: float = 5.0
    content_warnings: List[str] = field(default_factory=list)
    trigger_warnings: List[str] = field(default_factory=list)


@dataclass
class DifficultyFactors:
    """Difficulty assessment"""
    difficulty_level: float = 5.0  # 1-10
    entry_difficulty: float = 5.0
    sustained_difficulty: float = 5.0

    # Sources of difficulty
    difficulty_structural: float = 5.0
    difficulty_linguistic: float = 5.0
    difficulty_conceptual: float = 5.0
    difficulty_cultural: float = 5.0
    difficulty_emotional: float = 5.0


@dataclass
class Material:
    """
    Complete material model with 89+ parameters
    Represents a film, book, game, or series
    """
    # Core identification
    id: str
    metadata: MaterialMetadata

    # Physical characteristics
    physical: PhysicalParams = field(default_factory=PhysicalParams)

    # Content classification
    genre: GenreTaxonomy = field(default_factory=GenreTaxonomy)

    # Medium-specific
    visual: Optional[VisualCharacteristics] = None
    text: Optional[TextCharacteristics] = None
    audio: Optional[AudioCharacteristics] = None

    # Structure
    pacing: PacingStructure = field(default_factory=PacingStructure)
    characters: CharacterParameters = field(default_factory=CharacterParameters)

    # Content depth
    themes: ThematicParameters = field(default_factory=ThematicParameters)
    emotional: EmotionalParameters = field(default_factory=EmotionalParameters)

    # Tropes
    tropes: List[TropeUsage] = field(default_factory=list)

    # Cognitive aspects
    cognitive_ops: CognitiveOperations = field(default_factory=CognitiveOperations)
    transformative: TransformativePotential = field(default_factory=TransformativePotential)

    # Context
    cultural: CulturalContext = field(default_factory=CulturalContext)
    audience: AudienceParameters = field(default_factory=AudienceParameters)
    difficulty: DifficultyFactors = field(default_factory=DifficultyFactors)

    # Metadata
    added_date: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


# ============================================
# USER MODEL (multi-dimensional profile)
# ============================================

@dataclass
class UserDemographics:
    """Demographic data"""
    age: Optional[int] = None
    education_level: Optional[str] = None
    languages: List[str] = field(default_factory=lambda: ["en"])
    cultural_background: List[str] = field(default_factory=list)


@dataclass
class CognitiveProfile:
    """Cognitive profile"""
    primary_cognitive_style: CognitiveStyle = CognitiveStyle.ANALYTICAL
    secondary_styles: List[CognitiveStyle] = field(default_factory=list)
    style_flexibility: float = 0.5

    # Levels
    current_complexity_comfort: float = 5.0
    current_max_difficulty: float = 7.0
    growth_rate: float = 0.1

    # Meta-cognition
    meta_awareness_level: float = 5.0
    critical_thinking_skills: float = 5.0
    abstract_thinking: float = 5.0
    pattern_recognition: float = 5.0

    # Learning
    tolerance_for_ambiguity: float = 5.0
    need_for_closure: float = 5.0


@dataclass
class GenreLiteracy:
    """Genre knowledge"""
    genres: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cross_genre_understanding: float = 5.0


@dataclass
class TropeProfile:
    """Trope preferences and understanding"""
    trope_preferences: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    trope_fatigue: Dict[str, float] = field(default_factory=dict)
    trope_interest: Dict[str, float] = field(default_factory=dict)
    meta_trope_awareness: float = 5.0
    appreciation_for_meta: float = 5.0


@dataclass
class ThematicPreferences:
    """Thematic interests"""
    themes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    philosophical_interests: Dict[str, float] = field(default_factory=dict)

    # Worldview alignment
    worldview_optimism_pessimism: float = 0.0
    worldview_determinism_free_will: float = 0.0
    worldview_individualism_collectivism: float = 0.0

    # Challenge appetite
    belief_challenging: float = 5.0
    cognitive_dissonance_tolerance: float = 5.0
    paradigm_shift_openness: float = 5.0


@dataclass
class EmotionalProfile:
    """Emotional preferences and regulation"""
    preferred_emotions: List[str] = field(default_factory=list)
    avoided_emotions: List[str] = field(default_factory=list)
    emotional_intensity_preference: float = 5.0
    max_emotional_intensity: float = 8.0
    catharsis_seeking: float = 5.0
    emotional_resilience: float = 5.0

    # Triggers
    avoid_triggers: List[str] = field(default_factory=list)
    sensitive_topics: List[str] = field(default_factory=list)


@dataclass
class Rating:
    """User rating of a material"""
    material_id: str
    rating: float  # 0-10
    timestamp: datetime = field(default_factory=datetime.now)

    # Detailed feedback
    personal_transformation_score: float = 5.0
    difficulty_experienced: float = 5.0
    emotional_impact: float = 5.0
    intellectual_satisfaction: float = 5.0
    entertainment_value: float = 5.0

    # Context
    viewing_context: Optional[str] = None
    completion_rate: float = 1.0
    rewatch_probability: float = 0.5
    notes: str = ""


@dataclass
class GoalsMotivation:
    """User goals and motivation"""
    primary_goals: List[str] = field(default_factory=list)
    target_complexity_level: float = 7.0

    # Exploration strategy
    comfort_zone_preference: float = 5.0
    novelty_seeking: float = 5.0
    depth_vs_breadth: float = 0.0  # -1 depth, 1 breadth
    risk_tolerance: float = 5.0


@dataclass
class TimeResources:
    """Available time resources"""
    hours_per_week: float = 10.0
    session_length_preference: str = "medium"
    can_do_long_form: bool = True
    prefers_episodic: bool = False


@dataclass
class User:
    """
    Multi-dimensional user profile
    Captures preferences, cognitive style, and growth trajectory
    """
    # Core identification
    id: str
    username: str

    # Profile components
    demographics: UserDemographics = field(default_factory=UserDemographics)
    cognitive: CognitiveProfile = field(default_factory=CognitiveProfile)
    genre_literacy: GenreLiteracy = field(default_factory=GenreLiteracy)
    trope_profile: TropeProfile = field(default_factory=TropeProfile)
    thematic_prefs: ThematicPreferences = field(default_factory=ThematicPreferences)
    emotional: EmotionalProfile = field(default_factory=EmotionalProfile)
    goals: GoalsMotivation = field(default_factory=GoalsMotivation)
    time_resources: TimeResources = field(default_factory=TimeResources)

    # Interaction history
    ratings: List[Rating] = field(default_factory=list)
    viewed_materials: List[str] = field(default_factory=list)

    # Metadata
    created_date: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)


# ============================================
# TROPE MODEL
# ============================================

@dataclass
class Trope:
    """
    Trope definition with metadata
    """
    id: str
    name: str
    category: str
    description: str = ""
    transformation_potential: float = 5.0

    # Origins
    origin: Optional[str] = None
    popularized_by: Optional[str] = None
    academic_source: Optional[str] = None

    # Usage statistics
    usage_count: int = 0
    avg_execution_quality: float = 5.0


@dataclass
class TropeConnection:
    """
    Connection between materials through tropes
    """
    material_from: str
    material_to: str
    connection_type: str  # "influenced", "responded_to", "deconstructed"
    through_tropes: List[str] = field(default_factory=list)
    strength: float = 0.5
    notes: str = ""


@dataclass
class SessionContext:
    """
    Current session context
    """
    time_of_day: str = "evening"
    day_of_week: str = "weekday"
    device: str = "desktop"
    alone_or_with_others: str = "alone"

    # User state
    energy_level: float = 5.0
    focus_capacity: float = 5.0
    emotional_state: str = "neutral"
    stress_level: float = 5.0

    # Immediate needs
    need_relaxation: float = 5.0
    need_stimulation: float = 5.0
    need_escape: float = 5.0
    need_learning: float = 5.0
    need_inspiration: float = 5.0
