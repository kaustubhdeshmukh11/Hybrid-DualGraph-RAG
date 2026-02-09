"""
Entity-Based Knowledge Graph using NetworkX and Groq LLM
=========================================================
This module extracts entities and relationships from text using an LLM
and builds a knowledge graph with visualization capabilities.
"""

import os
import json
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# =============================================================================
# ENTITY & RELATIONSHIP EXTRACTION
# =============================================================================

EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from text.
Given the following text, extract:
1. ENTITIES: Important nouns (people, organizations, concepts, locations, objects, events)
2. RELATIONSHIPS: How entities are connected to each other

Return your response as a valid JSON object with this exact structure:
{{
    "entities": [
        {{"name": "entity_name", "type": "PERSON|ORGANIZATION|CONCEPT|LOCATION|EVENT|OBJECT"}}
    ],
    "relationships": [
        {{"source": "entity1", "relationship": "relationship_type", "target": "entity2"}}
    ]
}}

Rules:
- Entity names should be normalized (e.g., "John Smith" not "john smith" or "John")
- Relationship types should be short verb phrases (e.g., "works_at", "located_in", "is_a")
- Only extract clear, meaningful relationships
- Return ONLY the JSON, no other text

TEXT:
{text}
"""


def extract_entities_and_relationships(text: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> dict:
    """
    Extract entities and relationships from text using Groq LLM.
    
    Args:
        text: Input text to extract from
        model: Groq model to use
        
    Returns:
        Dictionary with 'entities' and 'relationships' lists
    """
    import re
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert knowledge graph builder. Always respond with valid JSON only. No markdown, no explanation, just pure JSON."
                },
                {
                    "role": "user", 
                    "content": EXTRACTION_PROMPT.format(text=text)
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        # Parse JSON response
        result = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        # Method 1: Direct parse
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Extract from markdown code block
        if "```" in result:
            # Find content between ``` markers
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', result)
            if code_match:
                try:
                    return json.loads(code_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Method 3: Find JSON object pattern
        json_match = re.search(r'\{[\s\S]*\}', result)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                print(f"JSON parsing error in Method 3: {e}")
                print(f"Attempted to parse: {json_match.group(0)[:100]}...")
                return {"entities": [], "relationships": []}
        
        print(f"Could not parse response: {result[:200]}...")
        return {"entities": [], "relationships": []}
        
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"entities": [], "relationships": []}


# =============================================================================
# KNOWLEDGE GRAPH CLASS
# =============================================================================

class KnowledgeGraph:
    """
    A NetworkX-based knowledge graph for storing entities and relationships.
    """
    
    # Color scheme for different entity types
    ENTITY_COLORS = {
        "PERSON": "#FF6B6B",        # Red
        "ORGANIZATION": "#4ECDC4",   # Teal
        "CONCEPT": "#45B7D1",        # Blue
        "LOCATION": "#96CEB4",       # Green
        "EVENT": "#FFEAA7",          # Yellow
        "OBJECT": "#DDA0DD",         # Plum
        "DEFAULT": "#C0C0C0"         # Gray
    }
    
    def __init__(self):
        """Initialize an empty knowledge graph."""
        self.graph = nx.DiGraph()  # Directed graph for relationships
        self.entity_types = {}     # Track entity types for visualization
        
    def add_entity(self, name: str, entity_type: str = "DEFAULT"):
        """
        Add an entity node to the graph.
        
        Args:
            name: Entity name
            entity_type: Type of entity (PERSON, ORGANIZATION, etc.)
        """
        self.graph.add_node(name, type=entity_type)
        self.entity_types[name] = entity_type.upper()
        
    def add_relationship(self, source: str, relationship: str, target: str):
        """
        Add a relationship edge between two entities.
        
        Args:
            source: Source entity name
            relationship: Relationship type/label
            target: Target entity name
        """
        # Ensure both entities exist
        if source not in self.graph:
            self.add_entity(source)
        if target not in self.graph:
            self.add_entity(target)
            
        self.graph.add_edge(source, target, relationship=relationship)
        
    def build_from_extraction(self, extraction: dict):
        """
        Build graph from extraction results.
        
        Args:
            extraction: Dictionary with 'entities' and 'relationships'
        """
        # Add entities
        for entity in extraction.get("entities", []):
            self.add_entity(
                name=entity["name"],
                entity_type=entity.get("type", "DEFAULT")
            )
            
        # Add relationships
        for rel in extraction.get("relationships", []):
            self.add_relationship(
                source=rel["source"],
                relationship=rel["relationship"],
                target=rel["target"]
            )
            
    def get_node_colors(self) -> list:
        """Get colors for all nodes based on their entity type."""
        colors = []
        for node in self.graph.nodes():
            entity_type = self.entity_types.get(node, "DEFAULT")
            colors.append(self.ENTITY_COLORS.get(entity_type, self.ENTITY_COLORS["DEFAULT"]))
        return colors
    
    def visualize(self, figsize: tuple = (14, 10), save_path: str = None):
        """
        Visualize the knowledge graph.
        
        Args:
            figsize: Figure size (width, height)
            save_path: Optional path to save the figure
        """
        if len(self.graph.nodes()) == 0:
            print("Graph is empty. Nothing to visualize.")
            return
            
        plt.figure(figsize=figsize, facecolor='#1a1a2e')
        ax = plt.gca()
        ax.set_facecolor('#1a1a2e')
        
        # Use spring layout for better visualization
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        
        # Draw nodes
        node_colors = self.get_node_colors()
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_color=node_colors,
            node_size=2500,
            alpha=0.9
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=9,
            font_weight='bold',
            font_color='white'
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph, pos,
            edge_color='#888888',
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1',
            alpha=0.7,
            width=2
        )
        
        # Draw edge labels (relationships)
        edge_labels = nx.get_edge_attributes(self.graph, 'relationship')
        nx.draw_networkx_edge_labels(
            self.graph, pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color='#AAAAAA',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#2d2d44', edgecolor='none', alpha=0.8)
        )
        
        # Create legend
        legend_elements = []
        for entity_type, color in self.ENTITY_COLORS.items():
            if entity_type != "DEFAULT":
                legend_elements.append(
                    plt.scatter([], [], c=color, s=150, label=entity_type)
                )
        
        plt.legend(
            handles=legend_elements,
            loc='upper left',
            facecolor='#2d2d44',
            edgecolor='white',
            labelcolor='white',
            fontsize=10
        )
        
        plt.title("Knowledge Graph", fontsize=16, color='white', fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none', bbox_inches='tight')
            print(f"Graph saved to: {save_path}")
            
        plt.show()
        
    def get_statistics(self) -> dict:
        """Get basic statistics about the graph."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "entity_types": dict(
                (t, list(self.entity_types.values()).count(t))
                for t in set(self.entity_types.values())
            ),
            "nodes": list(self.graph.nodes()),
            "relationships": [
                f"{u} --[{d['relationship']}]--> {v}"
                for u, v, d in self.graph.edges(data=True)
            ]
        }
        
    def query_neighbors(self, entity: str) -> dict:
        """
        Query all neighbors (incoming and outgoing) of an entity.
        
        Args:
            entity: Entity name to query
            
        Returns:
            Dictionary with incoming and outgoing relationships
        """
        if entity not in self.graph:
            return {"error": f"Entity '{entity}' not found in graph"}
            
        incoming = []
        outgoing = []
        
        # Outgoing edges
        for _, target, data in self.graph.out_edges(entity, data=True):
            outgoing.append({
                "target": target,
                "relationship": data.get("relationship", "related_to")
            })
            
        # Incoming edges
        for source, _, data in self.graph.in_edges(entity, data=True):
            incoming.append({
                "source": source,
                "relationship": data.get("relationship", "related_to")
            })
            
        return {
            "entity": entity,
            "type": self.entity_types.get(entity, "UNKNOWN"),
            "incoming": incoming,
            "outgoing": outgoing
        }


# =============================================================================
# MAIN FUNCTION - DEMO
# =============================================================================

def build_knowledge_graph(text: str, visualize: bool = True, save_path: str = None) -> KnowledgeGraph:
    """
    Build a knowledge graph from text.
    
    Args:
        text: Input text to process
        visualize: Whether to display the graph
        save_path: Optional path to save the visualization
        
    Returns:
        KnowledgeGraph object
    """
    print("=" * 60)
    print("KNOWLEDGE GRAPH BUILDER")
    print("=" * 60)
    
    # Step 1: Extract entities and relationships
    print("\n[1/3] Extracting entities and relationships using LLM...")
    extraction = extract_entities_and_relationships(text)
    
    print(f"    ✓ Found {len(extraction.get('entities', []))} entities")
    print(f"    ✓ Found {len(extraction.get('relationships', []))} relationships")
    
    # Step 2: Build the graph
    print("\n[2/3] Building knowledge graph...")
    kg = KnowledgeGraph()
    kg.build_from_extraction(extraction)
    
    # Step 3: Display statistics
    stats = kg.get_statistics()
    print(f"    ✓ Graph has {stats['total_nodes']} nodes and {stats['total_edges']} edges")
    print(f"\n    Entity Types:")
    for etype, count in stats.get('entity_types', {}).items():
        print(f"      - {etype}: {count}")
    
    print(f"\n    Relationships:")
    for rel in stats.get('relationships', []):
        print(f"      - {rel}")
    
    # Step 4: Visualize
    if visualize:
        print("\n[3/3] Generating visualization...")
        kg.visualize(save_path=save_path)
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH BUILT SUCCESSFULLY!")
    print("=" * 60)
    
    return kg


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example text for demonstration
    sample_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. 
    The company is headquartered in Cupertino, California. Tim Cook became the CEO 
    after Steve Jobs passed away in 2011. Apple develops products like the iPhone, 
    iPad, and MacBook. The company competes with Microsoft and Google in the 
    technology industry. Satya Nadella leads Microsoft, while Sundar Pichai is 
    the CEO of Google. Both companies are also based in the United States.
    """
    
    # Build and visualize the knowledge graph
    kg = build_knowledge_graph(
        text=sample_text,
        visualize=True,
        save_path="knowledge_graph.png"
    )
    
    # Example query
    print("\n" + "=" * 60)
    print("EXAMPLE QUERY: Apple Inc.")
    print("=" * 60)
    result = kg.query_neighbors("Apple Inc.")
    print(json.dumps(result, indent=2))
