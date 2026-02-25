#!/usr/bin/env python3
"""
Failure memory system — learn from what doesn't work.

Tracks failed attempts with context so future attempts can avoid same mistakes.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

REPO = Path.home() / "sybil-arc-challenge"
DB_PATH = REPO / "knowledge" / "patterns.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_failure_tables():
    """Add failure tracking tables to the database."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT NOT NULL,
            level INTEGER,
            attempt_number INTEGER DEFAULT 1,
            failure_mode TEXT,           -- 'timeout', 'wrong_strategy', 'missed_mechanic', etc.
            description TEXT,
            strategy_tried TEXT,         -- What approach was attempted
            actions_taken INTEGER,
            context JSON,                -- Additional context (frame snapshots, etc.)
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS failure_lessons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            failure_id INTEGER REFERENCES failures(id),
            lesson TEXT NOT NULL,        -- What we learned
            applies_to TEXT,             -- 'this_game', 'similar_games', 'all_games'
            confidence REAL DEFAULT 0.5,
            verified INTEGER DEFAULT 0,  -- 1 if lesson was verified useful
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_failures_game ON failures(game_id);
        CREATE INDEX IF NOT EXISTS idx_failures_mode ON failures(failure_mode);
    """)
    conn.commit()
    print("✅ Failure tracking tables initialized")

def record_failure(
    game_id: str,
    failure_mode: str,
    description: str,
    strategy_tried: str = "",
    level: int = None,
    attempt: int = 1,
    actions_taken: int = 0,
    context: dict = None
) -> int:
    """Record a failed attempt."""
    conn = get_db()
    
    cursor = conn.execute("""
        INSERT INTO failures (game_id, level, attempt_number, failure_mode, 
                            description, strategy_tried, actions_taken, context)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        game_id, level, attempt, failure_mode,
        description, strategy_tried, actions_taken,
        json.dumps(context) if context else None
    ))
    
    failure_id = cursor.lastrowid
    conn.commit()
    
    print(f"❌ Recorded failure #{failure_id}: {game_id} - {failure_mode}")
    return failure_id

def add_lesson(
    failure_id: int,
    lesson: str,
    applies_to: str = "this_game",
    confidence: float = 0.5
) -> int:
    """Add a lesson learned from a failure."""
    conn = get_db()
    
    cursor = conn.execute("""
        INSERT INTO failure_lessons (failure_id, lesson, applies_to, confidence)
        VALUES (?, ?, ?, ?)
    """, (failure_id, lesson, applies_to, confidence))
    
    lesson_id = cursor.lastrowid
    conn.commit()
    
    print(f"💡 Lesson #{lesson_id}: {lesson[:50]}...")
    return lesson_id

def verify_lesson(lesson_id: int, useful: bool):
    """Mark whether a lesson was actually useful."""
    conn = get_db()
    conn.execute("""
        UPDATE failure_lessons 
        SET verified = ?, confidence = confidence + ?
        WHERE id = ?
    """, (1 if useful else -1, 0.2 if useful else -0.1, lesson_id))
    conn.commit()

def get_failures_for_game(game_id: str) -> List[Dict]:
    """Get all failures for a specific game."""
    conn = get_db()
    rows = conn.execute("""
        SELECT f.*, GROUP_CONCAT(fl.lesson, '; ') as lessons
        FROM failures f
        LEFT JOIN failure_lessons fl ON f.id = fl.failure_id
        WHERE f.game_id = ?
        GROUP BY f.id
        ORDER BY f.timestamp DESC
    """, (game_id,)).fetchall()
    
    return [dict(r) for r in rows]

def get_lessons_for_game(game_id: str, min_confidence: float = 0.3) -> List[Dict]:
    """Get lessons that apply to a game."""
    conn = get_db()
    rows = conn.execute("""
        SELECT fl.*, f.game_id as source_game
        FROM failure_lessons fl
        JOIN failures f ON fl.failure_id = f.id
        WHERE (f.game_id = ? OR fl.applies_to IN ('similar_games', 'all_games'))
          AND fl.confidence >= ?
        ORDER BY fl.confidence DESC
    """, (game_id, min_confidence)).fetchall()
    
    return [dict(r) for r in rows]

def get_failure_context_prompt(game_id: str) -> str:
    """Generate context for an LLM about past failures on this game."""
    failures = get_failures_for_game(game_id)
    lessons = get_lessons_for_game(game_id)
    
    if not failures and not lessons:
        return ""
    
    prompt = f"\n=== FAILURE CONTEXT for {game_id} ===\n"
    
    if failures:
        prompt += f"\nPrevious failed attempts ({len(failures)}):\n"
        for f in failures[:5]:  # Limit to recent 5
            prompt += f"  - Attempt {f['attempt_number']}: {f['failure_mode']}\n"
            prompt += f"    Strategy: {f['strategy_tried']}\n"
            prompt += f"    Problem: {f['description']}\n"
    
    if lessons:
        prompt += f"\nLessons learned:\n"
        for l in lessons[:5]:
            conf = f"[{l['confidence']:.0%}]" if l['confidence'] else ""
            prompt += f"  - {conf} {l['lesson']}\n"
    
    prompt += "\nAvoid repeating these mistakes.\n"
    return prompt

def get_failure_stats() -> Dict:
    """Get overall failure statistics."""
    conn = get_db()
    
    total = conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
    by_mode = conn.execute("""
        SELECT failure_mode, COUNT(*) as count 
        FROM failures 
        GROUP BY failure_mode
    """).fetchall()
    lessons = conn.execute("SELECT COUNT(*) FROM failure_lessons").fetchone()[0]
    verified = conn.execute(
        "SELECT COUNT(*) FROM failure_lessons WHERE verified = 1"
    ).fetchone()[0]
    
    return {
        "total_failures": total,
        "by_mode": {r["failure_mode"]: r["count"] for r in by_mode},
        "total_lessons": lessons,
        "verified_lessons": verified
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python failure_memory.py <init|record|lessons|stats>")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "init":
        init_failure_tables()
    
    elif cmd == "stats":
        stats = get_failure_stats()
        print(json.dumps(stats, indent=2))
    
    elif cmd == "lessons" and len(sys.argv) > 2:
        game_id = sys.argv[2]
        lessons = get_lessons_for_game(game_id)
        for l in lessons:
            print(f"[{l['confidence']:.0%}] {l['lesson']}")
    
    elif cmd == "context" and len(sys.argv) > 2:
        game_id = sys.argv[2]
        print(get_failure_context_prompt(game_id))
    
    else:
        print("Commands: init, stats, lessons <game_id>, context <game_id>")
