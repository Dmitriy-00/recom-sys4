"""
Basic Usage Example for the Recommendation System
"""

from recom_system import (
    RecommendationSystem,
    Material, User, Trope,
    MaterialMetadata, MaterialType,
    TropeUsage, UsageType,
    SessionContext,
    Rating,
    CognitiveStyle
)
from recom_system.recommendation_engine import RecommendationEngine
from datetime import datetime


def create_sample_materials():
    """Create some sample materials for demonstration"""

    materials = []

    # Material 1: The Matrix (1999)
    matrix = Material(
        id="matrix_1999",
        metadata=MaterialMetadata(
            title="The Matrix",
            year=1999,
            creator="Wachowski Sisters",
            type=MaterialType.FILM
        )
    )

    # Set some parameters
    matrix.difficulty.difficulty_level = 6.0
    matrix.genre.primary_genre = "scifi"
    matrix.genre.genre_blend = {"scifi": 0.7, "action": 0.3}
    matrix.transformative.transformative_score = 8.5
    matrix.cultural.canonical_status = 0.9
    matrix.cultural.critical_acclaim = 8.7

    # Add tropes
    matrix.tropes = [
        TropeUsage(
            trope_id="chosen_one",
            trope_name="The Chosen One",
            category="character_archetype",
            usage_type=UsageType.STRAIGHT,
            awareness=9.0,
            execution=9.0,
            centrality=0.9
        ),
        TropeUsage(
            trope_id="reality_questioning",
            trope_name="Questioning Reality",
            category="philosophical_theme",
            usage_type=UsageType.STRAIGHT,
            awareness=9.0,
            execution=9.5,
            centrality=1.0,
            transformation_potential=9.0
        ),
        TropeUsage(
            trope_id="mentor_death",
            trope_name="Mentor's Death",
            category="plot_device",
            usage_type=UsageType.STRAIGHT,
            execution=8.0,
            centrality=0.6
        )
    ]

    # Themes
    matrix.themes.primary_themes = [
        {'theme': 'reality_vs_illusion', 'prominence': 1.0},
        {'theme': 'free_will', 'prominence': 0.8},
        {'theme': 'human_vs_machine', 'prominence': 0.7}
    ]
    matrix.themes.philosophical_depth_overall = 8.0
    matrix.themes.philosophical_depth_metaphysical = 9.0
    matrix.themes.worldview_determinism_free_will = 0.5  # Explores both

    materials.append(matrix)

    # Material 2: Inception (2010)
    inception = Material(
        id="inception_2010",
        metadata=MaterialMetadata(
            title="Inception",
            year=2010,
            creator="Christopher Nolan",
            type=MaterialType.FILM
        )
    )

    inception.difficulty.difficulty_level = 7.5
    inception.genre.primary_genre = "scifi"
    inception.genre.genre_blend = {"scifi": 0.6, "thriller": 0.4}
    inception.transformative.transformative_score = 7.5
    inception.cultural.canonical_status = 0.7
    inception.cultural.critical_acclaim = 8.4

    inception.tropes = [
        TropeUsage(
            trope_id="reality_questioning",
            trope_name="Questioning Reality",
            category="philosophical_theme",
            usage_type=UsageType.RECONSTRUCTION,
            awareness=9.0,
            execution=9.0,
            centrality=1.0,
            transformation_potential=8.0
        ),
        TropeUsage(
            trope_id="dream_within_dream",
            trope_name="Dream Within Dream",
            category="narrative_technique",
            usage_type=UsageType.STRAIGHT,
            execution=9.5,
            centrality=0.9
        ),
        TropeUsage(
            trope_id="heist",
            trope_name="The Heist",
            category="plot_device",
            usage_type=UsageType.STRAIGHT,
            execution=8.5,
            centrality=0.7
        )
    ]

    inception.themes.primary_themes = [
        {'theme': 'reality_vs_illusion', 'prominence': 1.0},
        {'theme': 'memory', 'prominence': 0.8},
        {'theme': 'loss_and_grief', 'prominence': 0.6}
    ]
    inception.themes.philosophical_depth_overall = 7.5

    materials.append(inception)

    # Material 3: Everything Everywhere All at Once (2022)
    eeaao = Material(
        id="eeaao_2022",
        metadata=MaterialMetadata(
            title="Everything Everywhere All at Once",
            year=2022,
            creator="Daniels",
            type=MaterialType.FILM
        )
    )

    eeaao.difficulty.difficulty_level = 8.0
    eeaao.genre.primary_genre = "scifi"
    eeaao.genre.genre_blend = {"scifi": 0.5, "comedy": 0.3, "drama": 0.2}
    eeaao.transformative.transformative_score = 9.0
    eeaao.cultural.canonical_status = 0.6
    eeaao.cultural.critical_acclaim = 8.9

    eeaao.tropes = [
        TropeUsage(
            trope_id="reality_questioning",
            trope_name="Questioning Reality",
            category="philosophical_theme",
            usage_type=UsageType.META,
            awareness=10.0,
            execution=9.5,
            centrality=1.0,
            transformation_potential=9.5,
            meta_commentary=True
        ),
        TropeUsage(
            trope_id="multiverse",
            trope_name="Multiverse",
            category="narrative_technique",
            usage_type=UsageType.STRAIGHT,
            execution=9.0,
            centrality=1.0
        ),
        TropeUsage(
            trope_id="chosen_one",
            trope_name="The Chosen One",
            category="character_archetype",
            usage_type=UsageType.DECONSTRUCTION,
            awareness=9.0,
            execution=8.5,
            centrality=0.5
        )
    ]

    eeaao.themes.primary_themes = [
        {'theme': 'nihilism_vs_meaning', 'prominence': 1.0},
        {'theme': 'family', 'prominence': 0.9},
        {'theme': 'identity', 'prominence': 0.7}
    ]
    eeaao.themes.philosophical_depth_overall = 8.5
    eeaao.themes.philosophical_depth_existential = 9.0

    materials.append(eeaao)

    return materials


def create_sample_user():
    """Create a sample user profile"""

    user = User(
        id="user_001",
        username="alice"
    )

    # Set cognitive profile
    user.cognitive.primary_cognitive_style = CognitiveStyle.ANALYTICAL
    user.cognitive.current_complexity_comfort = 6.0
    user.cognitive.current_max_difficulty = 8.0
    user.cognitive.meta_awareness_level = 7.0
    user.cognitive.tolerance_for_ambiguity = 7.5

    # Set genre literacy
    user.genre_literacy.genres = {
        'scifi': {
            'literacy_level': 8.0,
            'preference': 9.0,
            'exposure': 50
        },
        'drama': {
            'literacy_level': 6.0,
            'preference': 7.0,
            'exposure': 30
        }
    }

    # Set trope preferences (from past viewing)
    user.trope_profile.trope_preferences = {
        'reality_questioning_straight': {
            'rating': 9.0,
            'count': 5,
            'last_updated': datetime.now()
        },
        'chosen_one_straight': {
            'rating': 7.0,
            'count': 10,
            'last_updated': datetime.now()
        }
    }

    # Trope interests
    user.trope_profile.trope_interest = {
        'reality_questioning': 9.0,
        'multiverse': 8.0,
        'time_travel': 8.5
    }

    # Goals
    user.goals.primary_goals = ['intellectual_growth', 'entertainment']
    user.goals.novelty_seeking = 7.0
    user.goals.risk_tolerance = 6.5

    # Thematic preferences
    user.thematic_prefs.themes = {
        'reality_vs_illusion': {
            'interest_level': 9.0,
            'exposure': 10
        },
        'free_will': {
            'interest_level': 8.0,
            'exposure': 5
        }
    }
    user.thematic_prefs.belief_challenging = 7.0
    user.thematic_prefs.cognitive_dissonance_tolerance = 7.5

    # Emotional profile
    user.emotional.max_emotional_intensity = 8.0
    user.emotional.emotional_intensity_preference = 6.0

    # Add some ratings (The Matrix)
    user.ratings.append(Rating(
        material_id="matrix_1999",
        rating=9.0,
        personal_transformation_score=8.0,
        difficulty_experienced=6.0,
        emotional_impact=7.5,
        intellectual_satisfaction=9.0,
        notes="Mind-blowing! Changed how I think about reality."
    ))

    user.viewed_materials.append("matrix_1999")

    return user


def main():
    """Main demonstration"""

    print("=" * 60)
    print("TRANSFORMATIVE MEDIA RECOMMENDATION SYSTEM")
    print("=" * 60)
    print()

    # 1. Create recommendation engine
    print("1. Initializing recommendation engine...")
    engine = RecommendationEngine()

    # 2. Add materials
    print("2. Loading materials...")
    materials = create_sample_materials()
    for material in materials:
        engine.add_material(material)
        print(f"   - Added: {material.metadata.title} ({material.metadata.year})")

    # 3. Create user
    print("\n3. Creating user profile...")
    user = create_sample_user()
    engine.add_user(user)
    print(f"   - User: {user.username}")
    print(f"   - Current level: {user.cognitive.current_complexity_comfort}/10")
    print(f"   - Meta-awareness: {user.cognitive.meta_awareness_level}/10")
    print(f"   - Ratings: {len(user.ratings)}")

    # 4. Get recommendations
    print("\n4. Getting recommendations...")
    session = SessionContext(
        energy_level=7.0,
        emotional_state="curious",
        need_learning=8.0
    )

    recommendations = engine.get_recommendations(
        user_id="user_001",
        session_context=session,
        top_k=5
    )

    print(f"\n   Found {len(recommendations)} recommendations:\n")

    for i, rec in enumerate(recommendations, 1):
        material = rec['material']
        score = rec['final_score']
        confidence = rec['confidence']
        explanation = rec['explanation']
        badges = rec['badges']

        print(f"   {i}. {material.metadata.title} ({material.metadata.year})")
        print(f"      Score: {score:.2f}/10 (Confidence: {confidence:.2f})")
        print(f"      {explanation}")

        if badges:
            badge_str = " ".join([f"[{b['label']}]" for b in badges])
            print(f"      Badges: {badge_str}")

        # Show component breakdown
        print(f"      Components:")
        for comp_name, comp_data in rec['components'].items():
            comp_score = comp_data['score']
            weight = rec['weights'][comp_name]
            print(f"        - {comp_name}: {comp_score:.2f} (weight: {weight:.2%})")

        print()

    # 5. Analyze user profile
    print("\n5. User Profile Analysis:")
    analysis = engine.analyze_user_profile("user_001")
    print(f"   - Total ratings: {analysis['total_ratings']}")
    print(f"   - Average rating: {analysis['avg_rating']:.2f}")
    print(f"   - Current level: {analysis['current_level']}/10")
    print(f"   - Meta-awareness: {analysis['meta_awareness']}/10")
    print(f"   - Trope encounters: {analysis['trope_stats']['total_trope_encounters']}")

    if analysis['top_genres']:
        print(f"\n   Top genres:")
        for genre_data in analysis['top_genres']:
            print(f"      - {genre_data['genre']}: "
                  f"{genre_data['count']} materials, "
                  f"avg rating {genre_data['avg_rating']:.2f}")

    # 6. Exploration paths
    print("\n6. Exploration Paths (based on The Matrix):")
    paths = engine.get_exploration_paths("user_001", "matrix_1999")

    if paths:
        for path in paths[:3]:  # Show top 3
            print(f"\n   Path Type: {path['type']}")
            print(f"   Rationale: {path['rationale']}")
            print(f"   Transformation Score: {path.get('transformation_score', 0):.2f}")

            if 'materials' in path and path['materials']:
                print(f"   Recommended materials:")
                for mat_data in path['materials'][:3]:
                    mat = mat_data.get('material')
                    if mat:
                        print(f"      - {mat.metadata.title}")

    # 7. Knowledge gaps
    print("\n7. Knowledge Gaps:")
    gaps = engine.identify_knowledge_gaps("user_001")
    if gaps:
        print(f"   Found {len(gaps)} important tropes you haven't encountered:")
        for gap in gaps[:5]:
            trope = gap['trope']
            importance = gap['importance']
            print(f"      - {trope.name} (importance: {importance:.2f})")
            intro = gap.get('recommended_intro')
            if intro and 'material' in intro:
                print(f"        Best intro: {intro['material'].metadata.title}")

    # 8. Record a new rating
    print("\n8. Recording new rating...")
    new_rating = Rating(
        material_id="inception_2010",
        rating=8.5,
        personal_transformation_score=7.0,
        difficulty_experienced=7.5,
        intellectual_satisfaction=8.5,
        notes="Great complexity, loved the layered reality concept!"
    )

    engine.record_rating("user_001", "inception_2010", new_rating)
    print(f"   Rated Inception: {new_rating.rating}/10")
    print(f"   User trope preferences updated based on this rating")

    # 9. System statistics
    print("\n9. System Statistics:")
    stats = engine.get_system_statistics()
    print(f"   - Total materials: {stats['total_materials']}")
    print(f"   - Total users: {stats['total_users']}")
    print(f"   - Total ratings: {stats['total_ratings']}")

    trope_stats = stats.get('trope_map_stats', {})
    if trope_stats:
        print(f"   - Total tropes: {trope_stats.get('total_tropes', 0)}")
        print(f"   - Avg materials per trope: {trope_stats.get('avg_materials_per_trope', 0):.2f}")

    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
