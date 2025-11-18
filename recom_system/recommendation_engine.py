"""
Main Recommendation Engine
Combines all scoring components and implements the full algorithm
"""

from typing import List, Dict, Any, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

from .models import Material, User, SessionContext, Rating
from .trope_map import TropeMap
from .scoring_components import (
    TropeBasedScoring,
    ContentBasedScoring,
    StructuralBasedScoring,
    TransformationBasedScoring,
    CollaborativeFilteringScoring,
    PopularityBasedScoring,
    ContextAwareScoring,
    SerendipityScoring
)


class RecommendationEngine:
    """
    Complete recommendation engine implementing the full algorithm
    """

    def __init__(self, trope_map: Optional[TropeMap] = None):
        self.trope_map = trope_map or TropeMap()
        self.materials: Dict[str, Material] = {}
        self.users: Dict[str, User] = {}

        # Scoring components
        self.trope_scorer = TropeBasedScoring()
        self.content_scorer = ContentBasedScoring()
        self.structural_scorer = StructuralBasedScoring()
        self.transformation_scorer = TransformationBasedScoring()
        self.collaborative_scorer = CollaborativeFilteringScoring()
        self.popularity_scorer = PopularityBasedScoring()
        self.context_scorer = ContextAwareScoring()
        self.serendipity_scorer = SerendipityScoring()

        # Feedback buffer for online learning
        self.feedback_buffer: List[Rating] = []

    # ============================================
    # DATA MANAGEMENT
    # ============================================

    def add_material(self, material: Material):
        """Add material to the system"""
        self.materials[material.id] = material
        self.trope_map.add_material(material)

    def add_user(self, user: User):
        """Add user to the system"""
        self.users[user.id] = user

    def get_material(self, material_id: str) -> Optional[Material]:
        """Get material by ID"""
        return self.materials.get(material_id)

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)

    # ============================================
    # MAIN RECOMMENDATION METHOD
    # ============================================

    def get_recommendations(
        self,
        user_id: str,
        session_context: Optional[SessionContext] = None,
        top_k: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Main method to get recommendations for a user

        Args:
            user_id: User ID
            session_context: Current session context
            top_k: Number of recommendations to return
            filters: Optional filters (e.g., genre, year range)

        Returns:
            List of recommended materials with scores and explanations
        """

        user = self.get_user(user_id)
        if not user:
            return []

        if session_context is None:
            session_context = SessionContext()

        # Get candidate materials
        candidates = self._get_candidate_materials(user, filters)

        # Score all candidates
        scored_materials = []

        for material in candidates:
            result = self._score_material(material, user, session_context)
            scored_materials.append({
                'material': material,
                **result
            })

        # Sort by score
        scored_materials.sort(key=lambda x: x['final_score'], reverse=True)

        # Post-processing
        recommendations = self._post_process_recommendations(
            scored_materials,
            user,
            session_context,
            top_k
        )

        return recommendations

    def _get_candidate_materials(
        self,
        user: User,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Material]:
        """
        Get candidate materials for recommendation

        Filters out:
        - Already viewed materials (unless rewatchable)
        - Materials with triggers
        """

        candidates = []

        for material_id, material in self.materials.items():
            # Skip if already viewed (for now, simple check)
            if material_id in user.viewed_materials:
                # Could check rewatch_probability here
                continue

            # Apply filters
            if filters:
                if 'genre' in filters:
                    if material.genre.primary_genre not in filters['genre']:
                        continue

                if 'min_year' in filters:
                    if not material.metadata.year or material.metadata.year < filters['min_year']:
                        continue

                if 'max_year' in filters:
                    if not material.metadata.year or material.metadata.year > filters['max_year']:
                        continue

            candidates.append(material)

        return candidates

    def _score_material(
        self,
        material: Material,
        user: User,
        session_context: SessionContext
    ) -> Dict[str, Any]:
        """
        Score a single material using ensemble method
        """

        # 1. COMPUTE ALL COMPONENT SCORES
        components = {
            'trope_based': self.trope_scorer.score(material, user),
            'content_based': self.content_scorer.score(material, user),
            'structural_based': self.structural_scorer.score(material, user),
            'transformation_based': self.transformation_scorer.score(material, user),
            'collaborative': self.collaborative_scorer.score(
                material, user, list(self.users.values())
            ),
            'popularity': self.popularity_scorer.score(material),
            'context_aware': self.context_scorer.score(material, user, session_context),
            'serendipity': self.serendipity_scorer.score(material, user)
        }

        # 2. ADAPTIVE WEIGHTING
        weights = self._calculate_adaptive_weights(user, session_context)

        # 3. WEIGHTED ENSEMBLE
        weighted_score = sum(
            components[name]['score'] * weights[name]
            for name in components.keys()
        )

        # 4. CALIBRATION
        calibrated_score = self._calibrate_score(weighted_score, material, user)

        # 5. CONFIDENCE ESTIMATION
        confidence = self._estimate_confidence(components, user)

        return {
            'final_score': calibrated_score,
            'components': components,
            'weights': weights,
            'confidence': confidence
        }

    def _calculate_adaptive_weights(
        self,
        user: User,
        session_context: SessionContext
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights based on context

        Base weights:
        - Trope: 25%
        - Content: 20%
        - Structural: 20%
        - Transformation: 15%
        - Collaborative: 10%
        - Popularity: 5%
        - Context: 5%
        - Serendipity: 0% (added separately)
        """

        base_weights = {
            'trope_based': 0.25,
            'content_based': 0.20,
            'structural_based': 0.20,
            'transformation_based': 0.15,
            'collaborative': 0.10,
            'popularity': 0.05,
            'context_aware': 0.05,
            'serendipity': 0.00
        }

        # ADAPTATIONS

        # New user - more popularity
        if len(user.ratings) < 10:
            base_weights['popularity'] += 0.10
            base_weights['collaborative'] -= 0.05
            base_weights['trope_based'] -= 0.05

        # High energy - more structure/transformation
        if session_context.energy_level > 7:
            base_weights['structural_based'] += 0.05
            base_weights['transformation_based'] += 0.05
            base_weights['context_aware'] -= 0.10

        # Seeking discovery - more serendipity
        if session_context.need_inspiration > 7:
            base_weights['serendipity'] = 0.10
            base_weights['content_based'] -= 0.05
            base_weights['collaborative'] -= 0.05

        # Learning goals - more transformation
        if user.goals.primary_goals and 'intellectual_growth' in user.goals.primary_goals:
            base_weights['transformation_based'] += 0.10
            base_weights['popularity'] -= 0.05
            base_weights['context_aware'] -= 0.05

        # Normalize
        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()}

    def _calibrate_score(
        self,
        raw_score: float,
        material: Material,
        user: User
    ) -> float:
        """
        Calibrate score based on historical accuracy
        """

        # Simplified calibration
        # In production, would use past prediction errors

        # User rating bias
        if user.ratings:
            user_avg = np.mean([r.rating for r in user.ratings])
            global_avg = 6.5
            bias_correction = (user_avg - global_avg) * 0.3
            calibrated = raw_score + bias_correction
        else:
            calibrated = raw_score

        return max(0, min(10, calibrated))

    def _estimate_confidence(
        self,
        components: Dict[str, Dict[str, Any]],
        user: User
    ) -> float:
        """
        Estimate confidence in recommendation
        """

        confidences = []

        # Trope confidence
        if 'confidence' in components['trope_based']:
            confidences.append(components['trope_based']['confidence'])

        # Collaborative confidence
        if 'confidence' in components['collaborative']:
            confidences.append(components['collaborative']['confidence'])

        # User profile completeness
        profile_completeness = min(1.0, len(user.ratings) / 50)
        confidences.append(profile_completeness)

        return np.mean(confidences) if confidences else 0.5

    # ============================================
    # POST-PROCESSING
    # ============================================

    def _post_process_recommendations(
        self,
        scored_materials: List[Dict[str, Any]],
        user: User,
        session_context: SessionContext,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Post-process recommendations:
        - Safety filtering
        - Diversity optimization
        - ZPD prioritization
        - Explanation generation
        """

        # 1. SAFETY FILTERS
        safe_materials = self._apply_safety_filters(scored_materials, user)

        # 2. DIVERSITY OPTIMIZATION
        diverse_materials = self._optimize_diversity(safe_materials, k=top_k * 3)

        # 3. ZPD PRIORITIZATION
        zpd_boosted = self._boost_zpd_materials(diverse_materials, user)

        # 4. SELECT TOP K
        final_recommendations = zpd_boosted[:top_k]

        # 5. GENERATE EXPLANATIONS
        for rec in final_recommendations:
            rec['explanation'] = self._generate_explanation(rec, user)
            rec['badges'] = self._generate_badges(rec, user)

        return final_recommendations

    def _apply_safety_filters(
        self,
        materials: List[Dict[str, Any]],
        user: User
    ) -> List[Dict[str, Any]]:
        """Apply safety filters"""

        safe = []

        for item in materials:
            material = item['material']

            # Check triggers
            if any(trigger in user.emotional.avoid_triggers
                   for trigger in material.audience.trigger_warnings):
                continue

            # Check emotional intensity
            if material.emotional.emotional_intensity_overall > user.emotional.max_emotional_intensity:
                continue

            # Check difficulty
            if material.difficulty.difficulty_level > user.cognitive.current_max_difficulty + 1:
                continue

            safe.append(item)

        return safe

    def _optimize_diversity(
        self,
        materials: List[Dict[str, Any]],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Optimize diversity using Maximum Marginal Relevance (MMR)
        """

        if len(materials) <= k:
            return materials

        selected = []
        remaining = materials.copy()

        # First - most relevant
        selected.append(remaining.pop(0))

        lambda_param = 0.7  # Balance relevance vs diversity

        while len(selected) < k and remaining:
            best_score = -float('inf')
            best_idx = 0

            for idx, candidate in enumerate(remaining):
                # Relevance
                relevance = candidate['final_score']

                # Diversity (minimum similarity to selected)
                min_similarity = min(
                    self._calculate_similarity(candidate, selected_item)
                    for selected_item in selected
                )
                diversity = 1 - min_similarity

                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity * 10

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    def _calculate_similarity(
        self,
        item1: Dict[str, Any],
        item2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two materials
        """

        mat1 = item1['material']
        mat2 = item2['material']

        similarity = 0.0
        count = 0

        # Genre similarity
        if mat1.genre.primary_genre == mat2.genre.primary_genre:
            similarity += 1.0
        count += 1

        # Difficulty similarity
        diff_distance = abs(mat1.difficulty.difficulty_level - mat2.difficulty.difficulty_level)
        similarity += max(0, 1 - diff_distance / 10)
        count += 1

        # Trope overlap
        tropes1 = set(t.trope_id for t in mat1.tropes)
        tropes2 = set(t.trope_id for t in mat2.tropes)

        if tropes1 and tropes2:
            jaccard = len(tropes1 & tropes2) / len(tropes1 | tropes2)
            similarity += jaccard
            count += 1

        return similarity / count if count > 0 else 0.0

    def _boost_zpd_materials(
        self,
        materials: List[Dict[str, Any]],
        user: User
    ) -> List[Dict[str, Any]]:
        """
        Boost materials in Zone of Proximal Development
        """

        for item in materials:
            if item['components']['structural_based'].get('zpd_flag'):
                item['final_score'] *= 1.15  # 15% bonus
                item['zpd_boosted'] = True

        # Re-sort
        materials.sort(key=lambda x: x['final_score'], reverse=True)
        return materials

    def _generate_explanation(
        self,
        recommendation: Dict[str, Any],
        user: User
    ) -> str:
        """
        Generate human-readable explanation
        """

        explanation_parts = []
        components = recommendation['components']

        # ZPD flag (very important)
        if recommendation.get('zpd_boosted'):
            explanation_parts.append("⭐ В зоне ближайшего развития")

        # Strongest component
        component_scores = {
            name: data['score']
            for name, data in components.items()
        }

        strongest_name = max(component_scores.items(), key=lambda x: x[1])[0]
        strongest = components[strongest_name]

        # Add details from strongest component
        if 'details' in strongest and strongest['details']:
            explanation_parts.append(strongest['details'][0])

        # Transformation
        if components['transformation_based']['score'] >= 8:
            explanation_parts.append("Высокий трансформативный потенциал")

        # Context
        if components['context_aware']['score'] >= 7:
            explanation_parts.append("Подходит для текущего настроения")

        return " • ".join(explanation_parts[:3])

    def _generate_badges(
        self,
        recommendation: Dict[str, Any],
        user: User
    ) -> List[Dict[str, str]]:
        """
        Generate visual badges
        """

        badges = []

        # ZPD
        if recommendation.get('zpd_boosted'):
            badges.append({'type': 'zpd', 'label': 'Зона роста', 'color': 'gold'})

        # High transformation
        if recommendation['components']['transformation_based']['score'] >= 8.5:
            badges.append({'type': 'transform', 'label': 'Трансформация', 'color': 'purple'})

        # Perfect match
        if recommendation['final_score'] >= 9.0:
            badges.append({'type': 'match', 'label': 'Отличное соответствие', 'color': 'green'})

        # Canonical
        material = recommendation['material']
        if material.cultural.canonical_status > 0.8:
            badges.append({'type': 'canonical', 'label': 'Классика', 'color': 'blue'})

        # Challenge
        difficulty_diff = material.difficulty.difficulty_level - user.cognitive.current_complexity_comfort
        if difficulty_diff > 2:
            badges.append({'type': 'challenge', 'label': 'Вызов', 'color': 'orange'})

        return badges

    # ============================================
    # FEEDBACK & LEARNING
    # ============================================

    def record_rating(
        self,
        user_id: str,
        material_id: str,
        rating: Rating
    ):
        """
        Record user rating
        """

        user = self.get_user(user_id)
        if not user:
            return

        # Add to user's ratings
        user.ratings.append(rating)
        user.viewed_materials.append(material_id)

        # Add to feedback buffer
        self.feedback_buffer.append(rating)

        # Update trope preferences
        material = self.get_material(material_id)
        if material:
            self._update_trope_preferences(user, material, rating)

    def _update_trope_preferences(
        self,
        user: User,
        material: Material,
        rating: Rating
    ):
        """
        Update user's trope preferences based on rating
        """

        for trope_usage in material.tropes:
            key = f"{trope_usage.trope_id}_{trope_usage.usage_type.value}"

            if key not in user.trope_profile.trope_preferences:
                user.trope_profile.trope_preferences[key] = {
                    'rating': rating.rating,
                    'count': 1,
                    'last_updated': datetime.now()
                }
            else:
                pref = user.trope_profile.trope_preferences[key]
                # Moving average
                pref['rating'] = (pref['rating'] * pref['count'] + rating.rating) / (pref['count'] + 1)
                pref['count'] += 1
                pref['last_updated'] = datetime.now()

    # ============================================
    # EXPLORATION PATHS
    # ============================================

    def get_exploration_paths(
        self,
        user_id: str,
        liked_material_id: str
    ) -> List[Dict[str, Any]]:
        """
        Get personalized exploration paths based on liked material
        """

        user = self.get_user(user_id)
        if not user:
            return []

        return self.trope_map.build_exploration_path(
            user_current_level=user.cognitive.current_complexity_comfort,
            liked_material_id=liked_material_id,
            user_meta_awareness=user.cognitive.meta_awareness_level,
            user_cognitive_style=user.cognitive.primary_cognitive_style.value
        )

    def identify_knowledge_gaps(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Identify gaps in user's knowledge
        """

        user = self.get_user(user_id)
        if not user:
            return []

        # Find user's favorite genre
        genre_counts = defaultdict(int)
        for rating in user.ratings:
            material = self.get_material(rating.material_id)
            if material:
                genre_counts[material.genre.primary_genre] += 1

        if not genre_counts:
            return []

        favorite_genre = max(genre_counts.items(), key=lambda x: x[1])[0]

        return self.trope_map.identify_knowledge_gaps(
            user.viewed_materials,
            favorite_genre
        )

    # ============================================
    # ANALYTICS
    # ============================================

    def analyze_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Comprehensive user profile analysis
        """

        user = self.get_user(user_id)
        if not user:
            return {}

        # Rating statistics
        ratings = [r.rating for r in user.ratings]

        analysis = {
            'user_id': user_id,
            'total_ratings': len(ratings),
            'avg_rating': np.mean(ratings) if ratings else 0,
            'rating_variance': np.var(ratings) if ratings else 0,

            # Cognitive profile
            'current_level': user.cognitive.current_complexity_comfort,
            'max_difficulty': user.cognitive.current_max_difficulty,
            'meta_awareness': user.cognitive.meta_awareness_level,

            # Most consumed genres
            'top_genres': self._get_top_genres(user),

            # Trope profile
            'trope_stats': {
                'total_trope_encounters': len(user.trope_profile.trope_preferences),
                'fatigued_tropes': len(user.trope_profile.trope_fatigue),
                'interested_tropes': len(user.trope_profile.trope_interest)
            },

            # Knowledge gaps
            'knowledge_gaps_count': len(self.identify_knowledge_gaps(user_id))
        }

        return analysis

    def _get_top_genres(self, user: User, top_n: int = 5) -> List[Dict[str, Any]]:
        """Get user's most consumed genres"""

        genre_stats = defaultdict(lambda: {'count': 0, 'avg_rating': 0})

        for rating in user.ratings:
            material = self.get_material(rating.material_id)
            if material:
                genre = material.genre.primary_genre
                genre_stats[genre]['count'] += 1
                genre_stats[genre]['avg_rating'] = (
                    (genre_stats[genre]['avg_rating'] * (genre_stats[genre]['count'] - 1) +
                     rating.rating) / genre_stats[genre]['count']
                )

        sorted_genres = sorted(
            genre_stats.items(),
            key=lambda x: (x[1]['count'], x[1]['avg_rating']),
            reverse=True
        )

        return [
            {
                'genre': genre,
                'count': stats['count'],
                'avg_rating': stats['avg_rating']
            }
            for genre, stats in sorted_genres[:top_n]
        ]

    def get_system_statistics(self) -> Dict[str, Any]:
        """Get system-wide statistics"""

        return {
            'total_materials': len(self.materials),
            'total_users': len(self.users),
            'total_ratings': sum(len(user.ratings) for user in self.users.values()),
            'trope_map_stats': self.trope_map.get_map_statistics()
        }
