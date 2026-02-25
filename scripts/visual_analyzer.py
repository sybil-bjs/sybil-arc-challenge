#!/usr/bin/env python3
"""
Visual analysis for ARC-AGI-3 game frames.

Renders frames as images and can use vision models for analysis.
"""

import numpy as np
from pathlib import Path
import json

# ARC color palette (standard)
ARC_COLORS = {
    0: '#000000',  # black
    1: '#0074D9',  # blue
    2: '#FF4136',  # red
    3: '#2ECC40',  # green
    4: '#FFDC00',  # yellow
    5: '#AAAAAA',  # gray
    6: '#F012BE',  # magenta
    7: '#FF851B',  # orange
    8: '#7FDBFF',  # cyan
    9: '#870C25',  # brown
}

EMOJI_MAP = {
    0: '⬛', 1: '🔵', 2: '🔴', 3: '🟢', 4: '🟡',
    5: '⬜', 6: '🟣', 7: '🟠', 8: '🩵', 9: '🟤'
}

def frame_to_emoji(frame: np.ndarray) -> str:
    """Convert frame to emoji string for quick visualization."""
    lines = []
    for row in frame:
        line = ''.join(EMOJI_MAP.get(int(c), '?') for c in row)
        lines.append(line)
    return '\n'.join(lines)

def frame_to_image(frame: np.ndarray, output_path: str = None, scale: int = 8):
    """Convert frame to PNG image."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        
        # Create colormap from ARC colors
        colors = [ARC_COLORS[i] for i in range(10)]
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=(frame.shape[1]/8, frame.shape[0]/8), dpi=scale*8)
        ax.imshow(frame, cmap=cmap, vmin=0, vmax=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            return output_path
        else:
            # Return as bytes for inline display
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            return buf.getvalue()
    except ImportError:
        print("matplotlib not installed. Use frame_to_emoji instead.")
        return None

def compare_frames(frame1: np.ndarray, frame2: np.ndarray) -> dict:
    """Analyze what changed between two frames."""
    diff = frame1 != frame2
    changed_pixels = np.sum(diff)
    
    if changed_pixels == 0:
        return {"changed": False, "description": "No change"}
    
    # Find changed positions
    changed_positions = np.argwhere(diff)
    
    # Analyze the change
    changes = []
    for pos in changed_positions[:10]:  # First 10 changes
        y, x = pos
        old_val = int(frame1[y, x])
        new_val = int(frame2[y, x])
        changes.append({
            "position": (int(x), int(y)),
            "old_color": old_val,
            "new_color": new_val
        })
    
    # Detect movement (same color appeared elsewhere)
    colors_disappeared = set()
    colors_appeared = set()
    for c in changes:
        colors_disappeared.add(c["old_color"])
        colors_appeared.add(c["new_color"])
    
    # Heuristic: if same color disappeared and appeared, might be movement
    moved_colors = colors_disappeared & colors_appeared - {0}  # Ignore black (background)
    
    return {
        "changed": True,
        "changed_pixels": int(changed_pixels),
        "sample_changes": changes,
        "possible_movement": list(moved_colors) if moved_colors else None,
        "colors_appeared": list(colors_appeared - {0}),
        "colors_disappeared": list(colors_disappeared - {0})
    }

def analyze_frame(frame: np.ndarray) -> dict:
    """Analyze a single frame for objects and patterns."""
    analysis = {
        "size": frame.shape,
        "colors_present": [],
        "color_counts": {},
        "objects": []
    }
    
    # Count colors
    unique, counts = np.unique(frame, return_counts=True)
    for color, count in zip(unique, counts):
        color = int(color)
        if color != 0:  # Skip background
            analysis["colors_present"].append(color)
            analysis["color_counts"][color] = int(count)
    
    # Simple object detection: connected components per color
    try:
        from scipy import ndimage
        for color in analysis["colors_present"]:
            mask = frame == color
            labeled, num_features = ndimage.label(mask)
            if num_features > 0:
                for obj_id in range(1, num_features + 1):
                    obj_mask = labeled == obj_id
                    positions = np.argwhere(obj_mask)
                    if len(positions) > 0:
                        min_y, min_x = positions.min(axis=0)
                        max_y, max_x = positions.max(axis=0)
                        analysis["objects"].append({
                            "color": color,
                            "size": int(np.sum(obj_mask)),
                            "bbox": {
                                "x": int(min_x),
                                "y": int(min_y),
                                "width": int(max_x - min_x + 1),
                                "height": int(max_y - min_y + 1)
                            }
                        })
    except ImportError:
        pass  # scipy not available
    
    return analysis

def describe_frame_for_llm(frame: np.ndarray, previous_frame: np.ndarray = None) -> str:
    """Generate a text description of the frame for LLM consumption."""
    analysis = analyze_frame(frame)
    
    desc = f"Frame size: {analysis['size'][0]}x{analysis['size'][1]}\n"
    desc += f"Colors present: {analysis['colors_present']}\n"
    
    if analysis["objects"]:
        desc += f"Objects detected: {len(analysis['objects'])}\n"
        for i, obj in enumerate(analysis["objects"][:5]):
            desc += f"  - Object {i+1}: color={obj['color']}, size={obj['size']}px, "
            desc += f"at ({obj['bbox']['x']},{obj['bbox']['y']})\n"
    
    if previous_frame is not None:
        diff = compare_frames(previous_frame, frame)
        if diff["changed"]:
            desc += f"\nChanges from previous frame:\n"
            desc += f"  - {diff['changed_pixels']} pixels changed\n"
            if diff["possible_movement"]:
                desc += f"  - Possible movement of colors: {diff['possible_movement']}\n"
    
    return desc

if __name__ == "__main__":
    # Demo with a random frame
    demo_frame = np.random.randint(0, 10, size=(20, 20), dtype=np.int8)
    print("=== Emoji View ===")
    print(frame_to_emoji(demo_frame))
    print("\n=== Analysis ===")
    print(json.dumps(analyze_frame(demo_frame), indent=2))
