"""
Comprehensive Recommendation System for Transformative Media
Based on the algorithm from: Максимально полный алгоритм рекомендаций трансформативных медиа
"""

__version__ = "1.0.0"
__author__ = "Transformative Media Recommendations"

from .recommendation_engine import RecommendationEngine
from .models import (
    Material, User, Trope, TropeConnection,
    MaterialMetadata, MaterialType, UsageType,
    TropeUsage, SessionContext, Rating,
    CognitiveStyle
)
from .trope_map import TropeMap

# Alias for backward compatibility
RecommendationSystem = RecommendationEngine

__all__ = [
    'RecommendationEngine',
    'RecommendationSystem',
    'Material',
    'User',
    'Trope',
    'TropeConnection',
    'TropeMap',
    'MaterialMetadata',
    'MaterialType',
    'UsageType',
    'TropeUsage',
    'SessionContext',
    'Rating',
    'CognitiveStyle'
]
