"""
Scoring Components
Implements all 8 scoring components from the algorithm:
1. Trope-Based (25%)
2. Content-Based (20%)
3. Structural-Based (20%)
4. Transformation-Based (15%)
5. Collaborative Filtering (10%)
6. Popularity-Based (5%)
7. Context-Aware (5%)
8. Serendipity (5%)
"""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import numpy as np
import math
from datetime import datetime, timedelta

from .models import (
    Material, User, SessionContext, UsageType,
    CognitiveStyle, Rating
)


# ============================================
# COMPONENT 1: TROPE-BASED SCORING (25%)
# ============================================

class TropeBasedScoring:
    """
    Scores materials based on trope usage and user trope preferences
    Uses both explicit matching and embedding similarity
    """

    @staticmethod
    def score(material: Material, user: User) -> Dict[str, Any]:
        """
        Calculate trope-based score
        """
        score = 0.0
        details = []

        # 1. EXPLICIT TROPE MATCHING
        for trope_usage in material.tropes:
            key = f"{trope_usage.trope_id}_{trope_usage.usage_type.value}"

            # Check user preferences
            if key in user.trope_profile.trope_preferences:
                pref_data = user.trope_profile.trope_preferences[key]
                pref_rating = pref_data.get('rating', 5.0)

                # Weight by centrality and quality
                centrality_weight = trope_usage.centrality
                quality_weight = trope_usage.execution / 10.0

                # Time decay
                last_updated = pref_data.get('last_updated', datetime.now())
                time_decay = TropeBasedScoring._calculate_time_decay(last_updated)

                contribution = pref_rating * centrality_weight * quality_weight * time_decay * 0.3

                score += contribution

                if pref_rating >= 8:
                    details.append(f"Вам нравится {trope_usage.trope_name} ({trope_usage.usage_type.value})")

            # Check fatigue
            if trope_usage.trope_id in user.trope_profile.trope_fatigue:
                fatigue = user.trope_profile.trope_fatigue[trope_usage.trope_id]

                if trope_usage.usage_type == UsageType.STRAIGHT:
                    penalty = fatigue * trope_usage.centrality * 0.3
                    score -= penalty

                    if fatigue >= 7:
                        details.append(f"⚠️ Возможна усталость от {trope_usage.trope_name}")
                else:
                    # Deconstruction reduces fatigue
                    score += (fatigue / 10) * 0.2

        # 2. META-NARRATIVE MATCHING
        # Check if material's meta-level matches user's awareness
        material_meta_level = sum(
            1 for t in material.tropes
            if t.usage_type in [UsageType.DECONSTRUCTION, UsageType.META, UsageType.RECONSTRUCTION]
        ) / max(len(material.tropes), 1) * 10

        user_meta = user.cognitive.meta_awareness_level

        if abs(material_meta_level - user_meta) <= 1:
            score += 2.0
            if material_meta_level == user_meta + 1:
                details.append("⭐ Зона роста метаосознанности")
        elif material_meta_level > user_meta + 2:
            score -= 1.5
            details.append("Может быть слишком мета-нарративно")

        # 3. TROPE INTEREST BONUS
        for trope_usage in material.tropes:
            if trope_usage.trope_id in user.trope_profile.trope_interest:
                interest = user.trope_profile.trope_interest[trope_usage.trope_id]
                score += interest * trope_usage.centrality * 0.2

        # Normalize to 0-10
        final_score = min(10, max(0, score))

        return {
            'score': final_score,
            'details': details,
            'confidence': min(1.0, len(user.trope_profile.trope_preferences) / 20)
        }

    @staticmethod
    def _calculate_time_decay(last_updated: datetime) -> float:
        """Calculate time decay for preferences"""
        days_old = (datetime.now() - last_updated).days
        # Exponential decay with half-life of 180 days
        return math.exp(-days_old / 180)


# ============================================
# COMPONENT 2: CONTENT-BASED SCORING (20%)
# ============================================

class ContentBasedScoring:
    """
    Scores based on thematic, genre, and stylistic match
    """

    @staticmethod
    def score(material: Material, user: User) -> Dict[str, Any]:
        """
        Calculate content-based score
        """
        score = 0.0
        details = []

        # 1. THEMATIC MATCHING
        theme_score = ContentBasedScoring._score_themes(material, user)
        score += min(3.0, theme_score['score'])
        details.extend(theme_score['details'])

        # 2. GENRE MATCHING
        genre_score = ContentBasedScoring._score_genres(material, user)
        score += min(2.5, genre_score)

        # 3. CREATOR PREFERENCE
        if material.metadata.creator:
            # Check if we have data for this creator
            # Simplified: assume we track creator preferences
            score += 0.5  # Placeholder

        # 4. PHILOSOPHICAL ALIGNMENT
        philosophy_distance = ContentBasedScoring._calculate_worldview_distance(
            material.themes,
            user.thematic_prefs
        )

        if user.thematic_prefs.belief_challenging < 5:
            # Prefers aligned worldview
            score += (1 - philosophy_distance) * 1.5
        else:
            # Prefers challenge
            score += philosophy_distance * 1.5
            if philosophy_distance > 0.6:
                details.append("Бросает вызов вашим убеждениям")

        # 5. EMOTIONAL RESONANCE
        emotional_match = ContentBasedScoring._calculate_emotional_match(
            material.emotional,
            user.emotional
        )
        score += emotional_match * 0.5

        return {
            'score': min(10, score),
            'details': details
        }

    @staticmethod
    def _score_themes(material: Material, user: User) -> Dict[str, Any]:
        """Score thematic matching"""
        score = 0.0
        details = []

        for theme_data in material.themes.primary_themes:
            theme_id = theme_data.get('theme', '')
            prominence = theme_data.get('prominence', 0.5)

            if theme_id in user.thematic_prefs.themes:
                user_theme = user.thematic_prefs.themes[theme_id]
                interest = user_theme.get('interest_level', 5.0)

                contribution = prominence * (interest / 10) * 2.0
                score += contribution

                if interest >= 8:
                    details.append(f"Интересующая тема: {theme_id}")

        return {'score': score, 'details': details}

    @staticmethod
    def _score_genres(material: Material, user: User) -> float:
        """Score genre matching"""
        score = 0.0

        for genre, weight in material.genre.genre_blend.items():
            if genre in user.genre_literacy.genres:
                genre_data = user.genre_literacy.genres[genre]
                preference = genre_data.get('preference', 5.0)
                literacy = genre_data.get('literacy_level', 5.0)

                # Bonus for high literacy
                literacy_bonus = (literacy / 10) * 0.5

                contribution = (preference / 10) * weight * (1 + literacy_bonus)
                score += contribution

        return score

    @staticmethod
    def _calculate_worldview_distance(
        material_themes: Any,
        user_thematic: Any
    ) -> float:
        """Calculate Euclidean distance between worldviews"""
        dimensions = [
            'worldview_optimism_pessimism',
            'worldview_determinism_free_will',
            'worldview_individualism_collectivism',
            'worldview_idealism_pragmatism'
        ]

        distance_squared = 0.0
        for dim in dimensions:
            material_val = getattr(material_themes, dim, 0.0)
            user_val = getattr(user_thematic, dim, 0.0)
            distance_squared += (material_val - user_val) ** 2

        # Normalize to 0-1
        max_distance = len(dimensions) * 4  # max 2 per dimension
        return math.sqrt(distance_squared) / math.sqrt(max_distance)

    @staticmethod
    def _calculate_emotional_match(
        material_emotional: Any,
        user_emotional: Any
    ) -> float:
        """Calculate emotional resonance"""
        score = 0.0

        # Check intensity match
        material_intensity = material_emotional.emotional_intensity_overall
        user_preferred_intensity = user_emotional.emotional_intensity_preference

        intensity_diff = abs(material_intensity - user_preferred_intensity)
        if intensity_diff <= 2:
            score += 1.0
        elif intensity_diff > 4:
            score -= 0.5

        # Check if within acceptable range
        if material_intensity <= user_emotional.max_emotional_intensity:
            score += 0.5

        return score


# ============================================
# COMPONENT 3: STRUCTURAL-BASED SCORING (20%)
# ============================================

class StructuralBasedScoring:
    """
    Scores based on difficulty, pacing, and structural complexity
    Emphasizes Zone of Proximal Development (ZPD)
    """

    @staticmethod
    def score(material: Material, user: User) -> Dict[str, Any]:
        """
        Calculate structural-based score
        """
        score = 0.0
        details = []
        zpd_flag = False

        # 1. ZONE OF PROXIMAL DEVELOPMENT (Critical!)
        difficulty_diff = material.difficulty.difficulty_level - user.cognitive.current_complexity_comfort

        if 0 <= difficulty_diff <= 2:
            # IDEAL GROWTH ZONE
            zpd_score = 5.0

            if 0.5 <= difficulty_diff <= 1.5:
                zpd_score = 6.0
                zpd_flag = True
                details.append("⭐⭐ Идеальная зона роста!")
            elif difficulty_diff >= 1:
                details.append("⭐ Зона ближайшего развития")
                zpd_flag = True
            else:
                details.append("Комфортная сложность")

        elif -1 <= difficulty_diff < 0:
            zpd_score = 3.5
            details.append("Немного проще текущего уровня")

        elif 2 < difficulty_diff <= 3:
            zpd_score = 2.5
            if user.goals.risk_tolerance >= 7:
                zpd_score = 3.5
                details.append("Интересный вызов")
            else:
                details.append("⚠️ Может быть сложновато")

        elif difficulty_diff > user.cognitive.current_max_difficulty - user.cognitive.current_complexity_comfort:
            zpd_score = 0.5
            details.append("⚠️ Вероятно слишком сложно")

        else:
            zpd_score = 1.5
            if user.goals.comfort_zone_preference > 7:
                zpd_score = 3.0
            details.append("Может показаться простоватым")

        score += zpd_score

        # 2. DIFFICULTY SOURCES MATCH
        difficulty_match = StructuralBasedScoring._analyze_difficulty_match(
            material.difficulty,
            user.cognitive
        )
        score += difficulty_match * 2.0

        # 3. PACING MATCH
        # Simplified: assume user has pacing preference
        score += 1.0  # Placeholder

        # 4. COGNITIVE LOAD
        cognitive_load = StructuralBasedScoring._calculate_cognitive_load(material)
        user_capacity = user.cognitive.current_max_difficulty

        load_ratio = cognitive_load / user_capacity

        if 0.6 <= load_ratio <= 1.0:
            score += 1.5  # Optimal load
        elif load_ratio > 1.2:
            score -= 1.0
            details.append("Высокая когнитивная нагрузка")

        return {
            'score': min(10, max(0, score)),
            'details': details,
            'zpd_flag': zpd_flag
        }

    @staticmethod
    def _analyze_difficulty_match(
        material_difficulty: Any,
        user_cognitive: Any
    ) -> float:
        """Analyze if user's cognitive style matches difficulty sources"""

        style_strengths = {
            CognitiveStyle.VISUAL: {
                'difficulty_structural': 0.8,
                'difficulty_linguistic': 0.5,
                'difficulty_conceptual': 0.7
            },
            CognitiveStyle.ANALYTICAL: {
                'difficulty_structural': 1.0,
                'difficulty_linguistic': 0.7,
                'difficulty_conceptual': 1.0
            },
            CognitiveStyle.EMPATHETIC: {
                'difficulty_structural': 0.6,
                'difficulty_linguistic': 0.8,
                'difficulty_conceptual': 0.7
            },
            CognitiveStyle.KINESTHETIC: {
                'difficulty_structural': 0.7,
                'difficulty_linguistic': 0.6,
                'difficulty_conceptual': 0.6
            }
        }

        user_strengths = style_strengths[user_cognitive.primary_cognitive_style]
        match_score = 0.0
        count = 0

        for source in ['difficulty_structural', 'difficulty_linguistic', 'difficulty_conceptual']:
            difficulty = getattr(material_difficulty, source, 5.0)
            strength = user_strengths.get(source, 0.7)

            # Can handle?
            adjusted_difficulty = difficulty * strength
            if adjusted_difficulty <= user_cognitive.current_max_difficulty:
                match_score += 1.0
            elif adjusted_difficulty <= user_cognitive.current_max_difficulty + 1:
                match_score += 0.5
            else:
                match_score -= 0.5

            count += 1

        return match_score / count if count > 0 else 0.0

    @staticmethod
    def _calculate_cognitive_load(material: Material) -> float:
        """Calculate total cognitive load"""
        load = 0.0

        # Structural complexity
        load += (1 - material.pacing.linearity) * 2
        load += material.pacing.timeline_complexity * 0.5

        # Tropes
        for trope in material.tropes:
            load += trope.cognitive_load * trope.centrality * 0.1

        # Thematic density
        load += material.themes.thematic_density * 0.3

        # Philosophical depth
        load += material.themes.philosophical_depth_overall * 0.2

        return load


# ============================================
# COMPONENT 4: TRANSFORMATION-BASED (15%)
# ============================================

class TransformationBasedScoring:
    """
    Scores based on transformative potential and user readiness
    """

    @staticmethod
    def score(material: Material, user: User) -> Dict[str, Any]:
        """
        Calculate transformation-based score
        """
        base_score = material.transformative.transformative_score

        # 1. TROPE WORK BONUS
        meta_trope_count = sum(
            1 for t in material.tropes
            if t.usage_type in [UsageType.DECONSTRUCTION, UsageType.META, UsageType.RECONSTRUCTION]
        )

        if meta_trope_count >= 3:
            base_score += 0.5

        # 2. READINESS MULTIPLIER
        readiness = TransformationBasedScoring._calculate_readiness(
            material.transformative,
            user
        )

        adjusted_score = base_score * readiness

        # 3. GOAL ALIGNMENT
        # Simplified: check if transformation aligns with goals
        if 'intellectual_growth' in user.goals.primary_goals:
            adjusted_score += 1.0

        # 4. REWATCH VALUE BONUS
        if material.transformative.rewatch_value >= 8:
            adjusted_score += 0.5

        return {
            'score': min(10, adjusted_score),
            'readiness': readiness
        }

    @staticmethod
    def _calculate_readiness(
        transformative: Any,
        user: User
    ) -> float:
        """Calculate user readiness for transformation"""
        readiness = 1.0

        # Cognitive dissonance tolerance
        if transformative.cognitive_dissonance > 0.7:
            if user.cognitive.tolerance_for_ambiguity < 5:
                readiness *= 0.7
            else:
                readiness *= 1.2

        # Belief challenging
        if transformative.perspective_shifting > 0.7:
            if user.thematic_prefs.belief_challenging < 5:
                readiness *= 0.8
            else:
                readiness *= 1.3

        # Meta-cognitive readiness
        if transformative.metacognitive_awareness > 0.7:
            meta_diff = user.cognitive.meta_awareness_level - 7
            if meta_diff < 0:
                readiness *= (1 + meta_diff * 0.1)
            else:
                readiness *= 1.1

        # Emotional resilience
        if transformative.entry_barrier > user.emotional.max_emotional_intensity:
            readiness *= 0.6

        return max(0.5, min(1.5, readiness))


# ============================================
# COMPONENT 5: COLLABORATIVE FILTERING (10%)
# ============================================

class CollaborativeFilteringScoring:
    """
    User-based collaborative filtering
    """

    @staticmethod
    def score(
        material: Material,
        user: User,
        all_users: Optional[List[User]] = None
    ) -> Dict[str, Any]:
        """
        Calculate collaborative filtering score
        """

        if not all_users or len(all_users) < 10:
            return {'score': 5.0, 'confidence': 0.0}

        # Find similar users
        similar_users = CollaborativeFilteringScoring._find_similar_users(
            user,
            all_users,
            k=50
        )

        if not similar_users:
            return {'score': 5.0, 'confidence': 0.0}

        # Get ratings from similar users
        ratings = []
        similarities = []

        for similar_user, similarity in similar_users:
            # Check if this user rated the material
            for rating in similar_user.ratings:
                if rating.material_id == material.id:
                    ratings.append(rating.rating)
                    similarities.append(similarity)
                    break

        if not ratings:
            return {'score': 5.0, 'confidence': 0.0}

        # Weighted average
        weighted_avg = np.average(ratings, weights=similarities)

        # Confidence based on sample size
        confidence = min(1.0, len(ratings) / 10)

        return {
            'score': weighted_avg,
            'confidence': confidence
        }

    @staticmethod
    def _find_similar_users(
        user: User,
        all_users: List[User],
        k: int = 50
    ) -> List[Tuple[User, float]]:
        """Find k most similar users"""

        similarities = []

        for other_user in all_users:
            if other_user.id == user.id:
                continue

            if len(other_user.ratings) < 10:
                continue

            # Calculate similarity (simplified Pearson correlation)
            similarity = CollaborativeFilteringScoring._user_similarity(user, other_user)

            similarities.append((other_user, similarity))

        # Sort and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    @staticmethod
    def _user_similarity(user1: User, user2: User) -> float:
        """Calculate similarity between two users"""

        # Find common materials
        user1_ratings = {r.material_id: r.rating for r in user1.ratings}
        user2_ratings = {r.material_id: r.rating for r in user2.ratings}

        common = set(user1_ratings.keys()) & set(user2_ratings.keys())

        if len(common) < 3:
            return 0.0

        # Pearson correlation
        ratings1 = [user1_ratings[mid] for mid in common]
        ratings2 = [user2_ratings[mid] for mid in common]

        mean1 = np.mean(ratings1)
        mean2 = np.mean(ratings2)

        numerator = sum((r1 - mean1) * (r2 - mean2) for r1, r2 in zip(ratings1, ratings2))
        denominator = math.sqrt(
            sum((r1 - mean1) ** 2 for r1 in ratings1) *
            sum((r2 - mean2) ** 2 for r2 in ratings2)
        )

        if denominator == 0:
            return 0.0

        return max(0, numerator / denominator)  # Return 0-1


# ============================================
# COMPONENT 6: POPULARITY-BASED (5%)
# ============================================

class PopularityBasedScoring:
    """
    Scores based on critical acclaim and cultural impact
    """

    @staticmethod
    def score(material: Material) -> Dict[str, Any]:
        """
        Calculate popularity-based score
        """
        score = 0.0

        # Critical acclaim
        if material.cultural.critical_acclaim:
            score += (material.cultural.critical_acclaim / 10) * 2.0

        # Cultural impact
        if material.cultural.mainstream_penetration:
            score += (material.cultural.mainstream_penetration / 1.0) * 2.0

        # Canonical status
        if material.cultural.canonical_status:
            score += material.cultural.canonical_status * 2.5

        return {'score': min(10, score)}


# ============================================
# COMPONENT 7: CONTEXT-AWARE (5%)
# ============================================

class ContextAwareScoring:
    """
    Adjusts score based on current context and user state
    """

    @staticmethod
    def score(
        material: Material,
        user: User,
        context: SessionContext
    ) -> Dict[str, Any]:
        """
        Calculate context-aware score
        """
        score = 5.0  # Neutral base

        # 1. TIME OF DAY
        if context.time_of_day == 'late_night':
            if material.emotional.emotional_intensity_overall < 6:
                score += 1.5
            elif material.emotional.emotional_intensity_overall > 8:
                score -= 1.0

        # 2. ENERGY LEVEL
        if context.energy_level < 5:
            # Low energy - prefer easier materials
            adjusted_difficulty = material.difficulty.difficulty_level - 1.5
            if adjusted_difficulty <= user.cognitive.current_complexity_comfort:
                score += 1.5

        # 3. EMOTIONAL STATE
        emotional_match = ContextAwareScoring._match_mood(
            context.emotional_state,
            material.emotional
        )
        score += emotional_match * 1.0

        # 4. IMMEDIATE NEEDS
        if context.need_learning > 7:
            if material.transformative.transformative_score >= 7:
                score += 1.0

        if context.need_relaxation > 7:
            if material.emotional.emotional_intensity_overall < 5:
                score += 1.0

        return {'score': min(10, max(0, score))}

    @staticmethod
    def _match_mood(user_mood: str, material_emotional: Any) -> float:
        """Match user mood to material"""

        if user_mood in ['sad', 'melancholic']:
            if material_emotional.catharsis_potential > 7:
                return 1.5
            if material_emotional.final_tone > 0.5:
                return 1.0

        elif user_mood in ['stressed', 'anxious']:
            if material_emotional.emotional_intensity_overall < 5:
                return 1.5

        elif user_mood in ['energetic', 'excited']:
            if material_emotional.emotional_intensity_overall > 7:
                return 1.5

        return 0.0


# ============================================
# COMPONENT 8: SERENDIPITY (5%)
# ============================================

class SerendipityScoring:
    """
    Adds element of surprise and discovery
    """

    @staticmethod
    def score(material: Material, user: User, randomness: float = 0.05) -> Dict[str, Any]:
        """
        Calculate serendipity score
        """

        # Random boost
        random_boost = np.random.normal(0, 2.0) * randomness

        novelty_bonus = 0.0

        # 1. UNEXPLORED GENRE
        if material.genre.primary_genre not in user.genre_literacy.genres:
            novelty_bonus += 1.5

        # 2. UNKNOWN TROPES
        unknown_tropes = sum(
            1 for t in material.tropes
            if f"{t.trope_id}_{t.usage_type.value}" not in user.trope_profile.trope_preferences
        )
        if unknown_tropes >= 3:
            novelty_bonus += 1.0

        # 3. HIGH CANONICAL STATUS BUT NOT VIEWED
        if (material.cultural.canonical_status > 0.7 and
                material.id not in user.viewed_materials):
            novelty_bonus += 1.0

        # 4. BEYOND COMFORT ZONE
        if material.difficulty.difficulty_level > user.cognitive.current_max_difficulty:
            if user.goals.risk_tolerance > 6:
                novelty_bonus += 1.5

        score = 5.0 + random_boost + novelty_bonus

        return {'score': min(10, max(0, score))}
