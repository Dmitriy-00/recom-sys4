"""
Trope Map System
Implements the graph-based trope knowledge system from trope_map_documentation.md
"""

from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from .models import Material, Trope, TropeUsage, TropeConnection, UsageType


@dataclass
class TropeEvolutionStep:
    """
    One step in the evolution of a trope
    """
    material_id: str
    material_title: str
    year: int
    usage_type: UsageType
    execution_quality: float
    notes: str = ""


@dataclass
class TropeRelationship:
    """
    Relationship between two tropes
    """
    trope_a: str
    trope_b: str
    relationship: str  # "often_paired", "mutually_exclusive", "subverts"
    strength: float = 0.5
    examples: List[str] = field(default_factory=list)


class TropeMap:
    """
    Graph-based knowledge system for tropes

    Nodes: Tropes and Materials
    Edges: Usage relationships and influences
    """

    def __init__(self):
        # Main graph
        self.graph = nx.MultiDiGraph()

        # Indexed data
        self.tropes: Dict[str, Trope] = {}
        self.materials: Dict[str, Material] = {}
        self.trope_relationships: List[TropeRelationship] = []

        # Caches
        self._evolution_cache: Dict[str, List[TropeEvolutionStep]] = {}
        self._cluster_cache: Dict[str, Set[str]] = {}

    # ============================================
    # DATA MANAGEMENT
    # ============================================

    def add_trope(self, trope: Trope):
        """Add a trope to the map"""
        self.tropes[trope.id] = trope
        self.graph.add_node(
            trope.id,
            type='trope',
            name=trope.name,
            category=trope.category,
            transformation_potential=trope.transformation_potential
        )

    def add_material(self, material: Material):
        """Add a material to the map"""
        self.materials[material.id] = material
        self.graph.add_node(
            material.id,
            type='material',
            title=material.metadata.title,
            year=material.metadata.year,
            creator=material.metadata.creator
        )

        # Add edges for each trope usage
        for trope_usage in material.tropes:
            self._add_trope_usage_edge(material.id, trope_usage)

    def _add_trope_usage_edge(self, material_id: str, trope_usage: TropeUsage):
        """Add edge representing trope usage"""
        if trope_usage.trope_id not in self.tropes:
            # Create trope if it doesn't exist
            self.add_trope(Trope(
                id=trope_usage.trope_id,
                name=trope_usage.trope_name,
                category=trope_usage.category
            ))

        self.graph.add_edge(
            material_id,
            trope_usage.trope_id,
            usage_type=trope_usage.usage_type.value,
            awareness=trope_usage.awareness,
            execution=trope_usage.execution,
            centrality=trope_usage.centrality,
            transformation=trope_usage.transformation_potential
        )

        # Update trope statistics
        trope = self.tropes[trope_usage.trope_id]
        trope.usage_count += 1
        trope.avg_execution_quality = (
            (trope.avg_execution_quality * (trope.usage_count - 1) + trope_usage.execution)
            / trope.usage_count
        )

    def add_material_connection(self, connection: TropeConnection):
        """Add influence connection between materials"""
        self.graph.add_edge(
            connection.material_from,
            connection.material_to,
            type='influence',
            connection_type=connection.connection_type,
            through_tropes=connection.through_tropes,
            strength=connection.strength,
            notes=connection.notes
        )

    def add_trope_relationship(self, relationship: TropeRelationship):
        """Add relationship between tropes"""
        self.trope_relationships.append(relationship)
        self.graph.add_edge(
            relationship.trope_a,
            relationship.trope_b,
            type='trope_relationship',
            relationship=relationship.relationship,
            strength=relationship.strength,
            examples=relationship.examples
        )

    # ============================================
    # EXPLORATION PATHS
    # ============================================

    def build_exploration_path(
        self,
        user_current_level: float,
        liked_material_id: str,
        user_meta_awareness: float = 5.0,
        user_cognitive_style: str = "analytical"
    ) -> List[Dict[str, Any]]:
        """
        Build personalized exploration paths for a user
        Based on materials they liked

        Returns paths of type:
        - deepening: Same trope, higher complexity
        - expansion: Related tropes, same level
        - evolution: Historical trajectory
        - contrast: Alternative approaches
        """

        if liked_material_id not in self.materials:
            return []

        material = self.materials[liked_material_id]
        paths = []

        for trope_usage in material.tropes:
            if trope_usage.execution < 7:
                continue  # Only work with well-executed tropes

            trope_id = trope_usage.trope_id

            # 1. DEEPENING PATH
            deepening = self._build_deepening_path(
                trope_id,
                user_current_level,
                liked_material_id
            )
            if deepening:
                paths.append(deepening)

            # 2. EXPANSION PATH
            expansion = self._build_expansion_path(
                trope_id,
                user_current_level,
                material.genre.primary_genre
            )
            if expansion:
                paths.append(expansion)

            # 3. EVOLUTION PATH
            evolution = self._build_evolution_path(
                trope_id,
                material.metadata.year or 2000
            )
            if evolution:
                paths.append(evolution)

            # 4. CONTRAST PATH
            contrast = self._build_contrast_path(
                trope_id,
                trope_usage.usage_type
            )
            if contrast:
                paths.append(contrast)

        # Rank paths by transformative potential
        return self._rank_paths_by_transformation(
            paths,
            user_current_level,
            user_meta_awareness,
            user_cognitive_style
        )

    def _build_deepening_path(
        self,
        trope_id: str,
        user_level: float,
        exclude_material: str
    ) -> Optional[Dict[str, Any]]:
        """Find materials with same trope but more complex usage"""

        target_complexity = user_level + 1  # ZPD

        candidates = []
        for edge in self.graph.in_edges(trope_id, data=True):
            material_id = edge[0]
            data = edge[2]

            if material_id == exclude_material:
                continue

            if material_id not in self.materials:
                continue

            material = self.materials[material_id]

            # Check complexity
            if abs(material.difficulty.difficulty_level - target_complexity) <= 1:
                # Check for deconstruction/reconstruction/meta
                usage = data.get('usage_type', 'straight')
                if usage in ['deconstruction', 'reconstruction', 'meta']:
                    candidates.append({
                        'material_id': material_id,
                        'material': material,
                        'usage_type': usage,
                        'execution': data.get('execution', 5.0),
                        'transformation': data.get('transformation', 5.0)
                    })

        if not candidates:
            return None

        # Sort by execution quality
        candidates.sort(key=lambda x: x['execution'], reverse=True)

        return {
            'type': 'deepening',
            'trope': self.tropes[trope_id].name,
            'trope_id': trope_id,
            'materials': candidates[:5],
            'rationale': f"Explores {self.tropes[trope_id].name} more deeply through deconstruction/reconstruction"
        }

    def _build_expansion_path(
        self,
        trope_id: str,
        user_level: float,
        preferred_genre: str
    ) -> Optional[Dict[str, Any]]:
        """Find materials with related tropes"""

        related_tropes = self.get_often_paired_tropes(trope_id)
        if not related_tropes:
            return None

        candidates = []
        for related_id in related_tropes[:5]:  # Top 5 related
            for edge in self.graph.in_edges(related_id, data=True):
                material_id = edge[0]

                if material_id not in self.materials:
                    continue

                material = self.materials[material_id]

                # Same level, same genre preferred
                if (abs(material.difficulty.difficulty_level - user_level) <= 1 and
                        material.genre.primary_genre == preferred_genre):
                    candidates.append({
                        'material_id': material_id,
                        'material': material,
                        'related_trope': self.tropes[related_id].name,
                        'execution': edge[2].get('execution', 5.0)
                    })

        if not candidates:
            return None

        return {
            'type': 'expansion',
            'from_trope': self.tropes[trope_id].name,
            'to_tropes': [self.tropes[t].name for t in related_tropes[:5]],
            'materials': candidates[:5],
            'rationale': f"Expands understanding through related tropes"
        }

    def _build_evolution_path(
        self,
        trope_id: str,
        reference_year: int
    ) -> Optional[Dict[str, Any]]:
        """Show historical evolution of a trope"""

        evolution = self.get_trope_evolution(
            trope_id,
            start_year=reference_year - 20,
            end_year=reference_year + 10
        )

        if len(evolution) < 3:
            return None

        return {
            'type': 'evolution',
            'trope': self.tropes[trope_id].name,
            'trope_id': trope_id,
            'chain': evolution,
            'rationale': f"Shows how {self.tropes[trope_id].name} evolved over time"
        }

    def _build_contrast_path(
        self,
        trope_id: str,
        current_usage: UsageType
    ) -> Optional[Dict[str, Any]]:
        """Find materials that use trope differently"""

        # Look for opposite usage types
        contrast_types = {
            UsageType.STRAIGHT: [UsageType.DECONSTRUCTION, UsageType.SUBVERSION],
            UsageType.DECONSTRUCTION: [UsageType.RECONSTRUCTION, UsageType.META],
            UsageType.SUBVERSION: [UsageType.STRAIGHT, UsageType.RECONSTRUCTION],
            UsageType.RECONSTRUCTION: [UsageType.DECONSTRUCTION],
            UsageType.META: [UsageType.STRAIGHT, UsageType.RECONSTRUCTION]
        }

        target_types = contrast_types.get(current_usage, [])
        candidates = []

        for edge in self.graph.in_edges(trope_id, data=True):
            material_id = edge[0]
            data = edge[2]

            usage_str = data.get('usage_type', 'straight')
            try:
                usage = UsageType(usage_str)
                if usage in target_types:
                    if material_id in self.materials:
                        candidates.append({
                            'material_id': material_id,
                            'material': self.materials[material_id],
                            'usage_type': usage.value,
                            'execution': data.get('execution', 5.0)
                        })
            except ValueError:
                continue

        if not candidates:
            return None

        return {
            'type': 'contrast',
            'trope': self.tropes[trope_id].name,
            'trope_id': trope_id,
            'materials': candidates[:5],
            'rationale': f"Shows alternative approaches to {self.tropes[trope_id].name}"
        }

    def _rank_paths_by_transformation(
        self,
        paths: List[Dict[str, Any]],
        user_level: float,
        meta_awareness: float,
        cognitive_style: str
    ) -> List[Dict[str, Any]]:
        """Rank exploration paths by transformative potential"""

        for path in paths:
            score = 0.0

            # Deepening gets bonus for ZPD
            if path['type'] == 'deepening':
                score += 3.0

            # Deconstruction needs meta awareness
            if meta_awareness >= 7:
                for mat in path.get('materials', []):
                    if mat.get('usage_type') == 'deconstruction':
                        score += 2.0
                        break

            # Evolution for analytical thinkers
            if path['type'] == 'evolution' and cognitive_style == 'analytical':
                score += 2.0

            # Contrast always valuable
            if path['type'] == 'contrast':
                score += 1.0

            path['transformation_score'] = score

        # Sort by score
        return sorted(paths, key=lambda x: x.get('transformation_score', 0), reverse=True)

    # ============================================
    # TROPE ANALYSIS
    # ============================================

    def get_trope_evolution(
        self,
        trope_id: str,
        start_year: int = 1900,
        end_year: int = 2024
    ) -> List[TropeEvolutionStep]:
        """Get chronological evolution of a trope"""

        cache_key = f"{trope_id}_{start_year}_{end_year}"
        if cache_key in self._evolution_cache:
            return self._evolution_cache[cache_key]

        evolution = []

        for edge in self.graph.in_edges(trope_id, data=True):
            material_id = edge[0]
            data = edge[2]

            if material_id not in self.materials:
                continue

            material = self.materials[material_id]
            year = material.metadata.year

            if year and start_year <= year <= end_year:
                evolution.append(TropeEvolutionStep(
                    material_id=material_id,
                    material_title=material.metadata.title,
                    year=year,
                    usage_type=UsageType(data.get('usage_type', 'straight')),
                    execution_quality=data.get('execution', 5.0),
                    notes=f"Execution: {data.get('execution', 5.0)}/10"
                ))

        # Sort by year
        evolution.sort(key=lambda x: x.year)

        self._evolution_cache[cache_key] = evolution
        return evolution

    def get_often_paired_tropes(
        self,
        trope_id: str,
        min_strength: float = 0.5
    ) -> List[str]:
        """Find tropes often used together"""

        if trope_id in self._cluster_cache:
            return list(self._cluster_cache[trope_id])

        # Find materials using this trope
        materials_with_trope = set()
        for edge in self.graph.in_edges(trope_id):
            materials_with_trope.add(edge[0])

        # Count co-occurrences
        co_occurrence: Dict[str, int] = defaultdict(int)

        for material_id in materials_with_trope:
            if material_id not in self.materials:
                continue

            material = self.materials[material_id]
            for trope_usage in material.tropes:
                other_id = trope_usage.trope_id
                if other_id != trope_id:
                    co_occurrence[other_id] += 1

        # Calculate strength and filter
        total = len(materials_with_trope)
        if total == 0:
            return []

        paired = []
        for other_id, count in co_occurrence.items():
            strength = count / total
            if strength >= min_strength:
                paired.append((other_id, strength))

        # Sort by strength
        paired.sort(key=lambda x: x[1], reverse=True)

        result = [trope_id for trope_id, _ in paired]
        self._cluster_cache[trope_id] = set(result)
        return result

    def identify_knowledge_gaps(
        self,
        user_viewed_materials: List[str],
        favorite_genre: str
    ) -> List[Dict[str, Any]]:
        """
        Identify important tropes in a genre that user hasn't encountered
        """

        # Collect tropes user has seen
        consumed_tropes = set()
        for material_id in user_viewed_materials:
            if material_id in self.materials:
                material = self.materials[material_id]
                for trope_usage in material.tropes:
                    consumed_tropes.add(trope_usage.trope_id)

        # Find core tropes of genre
        core_tropes = self._get_core_genre_tropes(favorite_genre)

        # Identify gaps
        gaps = []
        for trope_id, importance in core_tropes:
            if trope_id not in consumed_tropes:
                # Find best introduction material
                intro_material = self._get_best_introduction(trope_id)

                gaps.append({
                    'trope': self.tropes[trope_id],
                    'importance': importance,
                    'recommended_intro': intro_material
                })

        # Sort by importance
        gaps.sort(key=lambda x: x['importance'], reverse=True)
        return gaps

    def _get_core_genre_tropes(self, genre: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """Get most important tropes for a genre"""

        trope_scores: Dict[str, float] = defaultdict(float)

        # Count usage in genre
        for material_id, material in self.materials.items():
            if material.genre.primary_genre == genre:
                for trope_usage in material.tropes:
                    # Weight by centrality and execution
                    score = trope_usage.centrality * (trope_usage.execution / 10)
                    trope_scores[trope_usage.trope_id] += score

        # Sort and return top N
        sorted_tropes = sorted(
            trope_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_tropes[:top_n]

    def _get_best_introduction(self, trope_id: str) -> Optional[Dict[str, Any]]:
        """Find best material to introduce a trope"""

        best_material = None
        best_score = -1

        for edge in self.graph.in_edges(trope_id, data=True):
            material_id = edge[0]
            data = edge[2]

            if material_id not in self.materials:
                continue

            material = self.materials[material_id]

            # Score = straight usage + high execution + moderate difficulty
            if data.get('usage_type') == 'straight':
                execution = data.get('execution', 0)
                difficulty = material.difficulty.difficulty_level

                # Prefer moderate difficulty (3-6) for introduction
                difficulty_score = 10 - abs(difficulty - 4.5)

                score = execution + difficulty_score

                if score > best_score:
                    best_score = score
                    best_material = {
                        'material_id': material_id,
                        'material': material,
                        'execution': execution,
                        'difficulty': difficulty
                    }

        return best_material

    # ============================================
    # ANALYSIS
    # ============================================

    def analyze_material_influences(self, material_id: str) -> Dict[str, Any]:
        """Analyze what influenced a material and what it influenced"""

        if material_id not in self.materials:
            return {}

        influenced_by = []
        influences_on = []

        # Find incoming influence edges
        for edge in self.graph.in_edges(material_id, data=True):
            if edge[2].get('type') == 'influence':
                influenced_by.append({
                    'material_id': edge[0],
                    'material': self.materials.get(edge[0]),
                    'connection_type': edge[2].get('connection_type'),
                    'through_tropes': edge[2].get('through_tropes', []),
                    'strength': edge[2].get('strength', 0.5)
                })

        # Find outgoing influence edges
        for edge in self.graph.out_edges(material_id, data=True):
            if edge[2].get('type') == 'influence':
                influences_on.append({
                    'material_id': edge[1],
                    'material': self.materials.get(edge[1]),
                    'connection_type': edge[2].get('connection_type'),
                    'through_tropes': edge[2].get('through_tropes', []),
                    'strength': edge[2].get('strength', 0.5)
                })

        return {
            'influenced_by': influenced_by,
            'influences_on': influences_on
        }

    def get_map_statistics(self) -> Dict[str, Any]:
        """Get statistics about the trope map"""

        total_tropes = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'trope'])
        total_materials = len([n for n, d in self.graph.nodes(data=True) if d.get('type') == 'material'])

        # Usage type distribution
        usage_types: Dict[str, int] = defaultdict(int)
        for edge in self.graph.edges(data=True):
            usage = edge[2].get('usage_type')
            if usage:
                usage_types[usage] += 1

        # Materials per trope
        trope_material_counts = []
        for trope_id in self.tropes.keys():
            count = len(list(self.graph.in_edges(trope_id)))
            trope_material_counts.append(count)

        avg_materials_per_trope = (
            sum(trope_material_counts) / len(trope_material_counts)
            if trope_material_counts else 0
        )

        return {
            'total_tropes': total_tropes,
            'total_materials': total_materials,
            'usage_type_distribution': dict(usage_types),
            'avg_materials_per_trope': avg_materials_per_trope,
            'tropes_with_5plus_materials': sum(1 for c in trope_material_counts if c >= 5)
        }
