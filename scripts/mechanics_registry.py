"""
Mechanics Registry for ARC-AGI-3

Persistent knowledge base of game mechanics that grows over time.
SemanticDiscovery uses this to know what to probe for in new games.

Usage:
    registry = MechanicsRegistry()
    registry.load()
    
    for mechanic in registry.get_by_priority('high'):
        discovery.probe_for(mechanic)

Saber ⚔️ | Bridget 🎮 | Sybil 🔬
2026-02-28
"""

import os
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DetectionRule:
    """A single rule for detecting a mechanic."""
    rule_type: str  # visual, behavioral
    condition: str
    threshold: Optional[str] = None
    required: bool = False
    note: Optional[str] = None


@dataclass  
class Mechanic:
    """A discovered game mechanic."""
    name: str
    description: str
    first_seen: str
    confidence: str
    
    # Detection rules
    visual_rules: List[DetectionRule] = field(default_factory=list)
    behavioral_rules: List[DetectionRule] = field(default_factory=list)
    
    # Interaction
    trigger: str = "overlap"
    effect: str = ""
    consumable: bool = False
    
    # Strategy
    strategy_implications: List[str] = field(default_factory=list)
    human_intuition: str = ""
    
    # Examples
    examples: List[Dict] = field(default_factory=list)
    
    # Meta
    priority: str = "medium"
    games_seen: List[str] = field(default_factory=list)


class MechanicsRegistry:
    """
    Loads and manages the mechanics knowledge base.
    
    The registry grows as we encounter new game mechanics.
    Each mechanic has detection rules that SemanticDiscovery uses.
    """
    
    def __init__(self, knowledge_dir: Optional[str] = None):
        if knowledge_dir is None:
            # Default to project's knowledge/mechanics directory
            script_dir = Path(__file__).parent
            knowledge_dir = script_dir.parent / "knowledge" / "mechanics"
        
        self.knowledge_dir = Path(knowledge_dir)
        self.mechanics: Dict[str, Mechanic] = {}
        self.registry_index: Dict = {}
    
    def load(self) -> bool:
        """Load all mechanics from the registry."""
        registry_file = self.knowledge_dir / "registry.yaml"
        
        if not registry_file.exists():
            print(f"⚠️ Registry not found at {registry_file}")
            return False
        
        # Load index
        with open(registry_file, 'r') as f:
            self.registry_index = yaml.safe_load(f)
        
        # Load each mechanic
        for entry in self.registry_index.get('mechanics', []):
            mechanic_file = self.knowledge_dir / entry['file']
            if mechanic_file.exists():
                mechanic = self._load_mechanic(mechanic_file, entry)
                if mechanic:
                    self.mechanics[mechanic.name] = mechanic
                    print(f"  ✓ Loaded mechanic: {mechanic.name}")
            else:
                print(f"  ⚠️ Mechanic file not found: {mechanic_file}")
        
        print(f"\n📚 Loaded {len(self.mechanics)} mechanics from registry")
        return True
    
    def _load_mechanic(self, filepath: Path, index_entry: Dict) -> Optional[Mechanic]:
        """Load a single mechanic from YAML file."""
        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            
            # Parse detection rules
            visual_rules = []
            behavioral_rules = []
            
            detection = data.get('detection', {})
            
            for rule in detection.get('visual', []):
                if isinstance(rule, dict):
                    visual_rules.append(DetectionRule(
                        rule_type='visual',
                        condition=rule.get('type', ''),
                        threshold=rule.get('threshold'),
                        note=rule.get('note'),
                    ))
            
            for rule in detection.get('behavioral', []):
                if isinstance(rule, dict):
                    behavioral_rules.append(DetectionRule(
                        rule_type='behavioral',
                        condition=rule.get('condition', ''),
                        required=rule.get('required', False),
                        note=rule.get('note'),
                    ))
            
            # Parse interaction
            interaction = data.get('interaction', {})
            
            # Parse strategy
            strategy = data.get('strategy_implications', [])
            if isinstance(strategy, str):
                strategy = [strategy]
            
            return Mechanic(
                name=data.get('name', ''),
                description=data.get('description', ''),
                first_seen=data.get('first_seen', ''),
                confidence=data.get('confidence', 'low'),
                visual_rules=visual_rules,
                behavioral_rules=behavioral_rules,
                trigger=interaction.get('trigger', 'overlap'),
                effect=interaction.get('effect', ''),
                consumable=interaction.get('consumable', False),
                strategy_implications=strategy,
                human_intuition=data.get('human_intuition', ''),
                examples=data.get('examples', []),
                priority=index_entry.get('priority', 'medium'),
                games_seen=index_entry.get('games_seen', []),
            )
            
        except Exception as e:
            print(f"  ❌ Error loading {filepath}: {e}")
            return None
    
    def get_all(self) -> List[Mechanic]:
        """Get all loaded mechanics."""
        return list(self.mechanics.values())
    
    def get_by_priority(self, priority: str) -> List[Mechanic]:
        """Get mechanics filtered by priority (high, medium, low)."""
        return [m for m in self.mechanics.values() if m.priority == priority]
    
    def get_by_name(self, name: str) -> Optional[Mechanic]:
        """Get a specific mechanic by name."""
        return self.mechanics.get(name)
    
    def get_detection_checklist(self) -> List[Dict]:
        """
        Get a checklist of things to probe for during discovery.
        
        Returns list of detection tasks sorted by priority.
        """
        checklist = []
        
        # High priority first
        for priority in ['high', 'medium', 'low']:
            for mechanic in self.get_by_priority(priority):
                for rule in mechanic.visual_rules:
                    checklist.append({
                        'mechanic': mechanic.name,
                        'type': 'visual',
                        'check': rule.condition,
                        'threshold': rule.threshold,
                        'priority': priority,
                    })
                
                for rule in mechanic.behavioral_rules:
                    if rule.required:
                        checklist.append({
                            'mechanic': mechanic.name,
                            'type': 'behavioral',
                            'check': rule.condition,
                            'priority': priority,
                        })
        
        return checklist
    
    def add_mechanic(self, mechanic: Mechanic, save: bool = True) -> bool:
        """
        Add a new mechanic to the registry.
        
        Call this when discovering new mechanics during gameplay.
        """
        self.mechanics[mechanic.name] = mechanic
        
        if save:
            # Save to YAML file
            filepath = self.knowledge_dir / f"{mechanic.name}.yaml"
            self._save_mechanic(mechanic, filepath)
            
            # Update registry index
            self._update_registry_index(mechanic)
        
        print(f"📝 Added new mechanic: {mechanic.name}")
        return True
    
    def _save_mechanic(self, mechanic: Mechanic, filepath: Path):
        """Save a mechanic to YAML file."""
        data = {
            'name': mechanic.name,
            'description': mechanic.description,
            'first_seen': mechanic.first_seen,
            'confidence': mechanic.confidence,
            'detection': {
                'visual': [
                    {'type': r.condition, 'threshold': r.threshold, 'note': r.note}
                    for r in mechanic.visual_rules
                ],
                'behavioral': [
                    {'condition': r.condition, 'required': r.required, 'note': r.note}
                    for r in mechanic.behavioral_rules
                ],
            },
            'interaction': {
                'trigger': mechanic.trigger,
                'effect': mechanic.effect,
                'consumable': mechanic.consumable,
            },
            'strategy_implications': mechanic.strategy_implications,
            'human_intuition': mechanic.human_intuition,
            'examples': mechanic.examples,
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def _update_registry_index(self, mechanic: Mechanic):
        """Update the registry.yaml index with new mechanic."""
        registry_file = self.knowledge_dir / "registry.yaml"
        
        # Add to index
        if 'mechanics' not in self.registry_index:
            self.registry_index['mechanics'] = []
        
        # Check if already exists
        existing = [m for m in self.registry_index['mechanics'] 
                   if m['name'] == mechanic.name]
        
        if not existing:
            self.registry_index['mechanics'].append({
                'name': mechanic.name,
                'file': f"{mechanic.name}.yaml",
                'priority': mechanic.priority,
                'games_seen': mechanic.games_seen,
            })
            
            self.registry_index['total_mechanics'] = len(self.registry_index['mechanics'])
            
            with open(registry_file, 'w') as f:
                yaml.dump(self.registry_index, f, default_flow_style=False, sort_keys=False)


def test_registry():
    """Test loading the mechanics registry."""
    print("\n🔬 Testing Mechanics Registry")
    print("=" * 40)
    
    registry = MechanicsRegistry()
    registry.load()
    
    print("\n📋 Detection Checklist:")
    for item in registry.get_detection_checklist()[:10]:
        print(f"  [{item['priority']}] {item['mechanic']}: {item['check']}")
    
    print("\n🎯 High Priority Mechanics:")
    for m in registry.get_by_priority('high'):
        print(f"  - {m.name}: {m.description[:60]}...")


if __name__ == "__main__":
    test_registry()
