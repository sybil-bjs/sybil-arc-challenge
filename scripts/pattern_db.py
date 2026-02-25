#!/usr/bin/env python3
"""
Pattern Knowledge Base for ARC-AGI solving.

Stores learned patterns with metadata for retrieval.

Usage:
  python pattern_db.py init                    # Create database
  python pattern_db.py add <name> [options]    # Add a pattern
  python pattern_db.py search <query>          # Search patterns
  python pattern_db.py list                    # List all patterns
  python pattern_db.py stats                   # Show statistics
  python pattern_db.py record <task_id> <pattern_name> <success|fail>
"""

import sqlite3
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

REPO = Path.home() / "sybil-arc-challenge"
DB_PATH = REPO / "knowledge" / "patterns.db"

def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        -- Core patterns table
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            keywords TEXT,  -- comma-separated
            code_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Pattern usage history
        CREATE TABLE IF NOT EXISTS pattern_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER REFERENCES patterns(id),
            task_id TEXT NOT NULL,
            success INTEGER NOT NULL,  -- 1 = success, 0 = failure
            notes TEXT,
            timestamp TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        -- Visual features for matching
        CREATE TABLE IF NOT EXISTS pattern_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_id INTEGER REFERENCES patterns(id),
            feature_type TEXT NOT NULL,  -- e.g., 'has_marker', 'shape_type', 'transform'
            feature_value TEXT NOT NULL
        );
        
        -- Full-text search index
        CREATE VIRTUAL TABLE IF NOT EXISTS patterns_fts USING fts5(
            name, description, keywords,
            content='patterns',
            content_rowid='id'
        );
        
        -- Triggers to keep FTS in sync
        CREATE TRIGGER IF NOT EXISTS patterns_ai AFTER INSERT ON patterns BEGIN
            INSERT INTO patterns_fts(rowid, name, description, keywords)
            VALUES (new.id, new.name, new.description, new.keywords);
        END;
        
        CREATE TRIGGER IF NOT EXISTS patterns_ad AFTER DELETE ON patterns BEGIN
            INSERT INTO patterns_fts(patterns_fts, rowid, name, description, keywords)
            VALUES ('delete', old.id, old.name, old.description, old.keywords);
        END;
        
        CREATE TRIGGER IF NOT EXISTS patterns_au AFTER UPDATE ON patterns BEGIN
            INSERT INTO patterns_fts(patterns_fts, rowid, name, description, keywords)
            VALUES ('delete', old.id, old.name, old.description, old.keywords);
            INSERT INTO patterns_fts(rowid, name, description, keywords)
            VALUES (new.id, new.name, new.description, new.keywords);
        END;
    """)
    conn.commit()
    print(f"✅ Database initialized at {DB_PATH}")
    return conn

def add_pattern(name: str, description: str = "", keywords: str = "", 
                code_path: str = "", features: list = None):
    conn = get_db()
    try:
        conn.execute("""
            INSERT INTO patterns (name, description, keywords, code_path)
            VALUES (?, ?, ?, ?)
        """, (name, description, keywords, code_path))
        
        pattern_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Add features if provided
        if features:
            for feat_type, feat_value in features:
                conn.execute("""
                    INSERT INTO pattern_features (pattern_id, feature_type, feature_value)
                    VALUES (?, ?, ?)
                """, (pattern_id, feat_type, feat_value))
        
        conn.commit()
        print(f"✅ Added pattern: {name} (id={pattern_id})")
        return pattern_id
    except sqlite3.IntegrityError:
        print(f"⚠️ Pattern '{name}' already exists")
        return None

def search_patterns(query: str, limit: int = 10):
    conn = get_db()
    
    # Full-text search
    results = conn.execute("""
        SELECT p.*, 
               bm25(patterns_fts) as score,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id AND pu.success = 1) as successes,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id) as total_uses
        FROM patterns_fts fts
        JOIN patterns p ON fts.rowid = p.id
        WHERE patterns_fts MATCH ?
        ORDER BY score
        LIMIT ?
    """, (query, limit)).fetchall()
    
    if not results:
        # Fallback to LIKE search
        results = conn.execute("""
            SELECT p.*,
                   0 as score,
                   (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id AND pu.success = 1) as successes,
                   (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id) as total_uses
            FROM patterns p
            WHERE p.name LIKE ? OR p.description LIKE ? OR p.keywords LIKE ?
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()
    
    return results

def search_by_features(features: dict, limit: int = 10):
    """Search patterns by visual features."""
    conn = get_db()
    
    # Build query for feature matching
    conditions = []
    params = []
    for feat_type, feat_value in features.items():
        conditions.append("(pf.feature_type = ? AND pf.feature_value = ?)")
        params.extend([feat_type, feat_value])
    
    if not conditions:
        return []
    
    query = f"""
        SELECT p.*, COUNT(DISTINCT pf.feature_type) as match_count,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id AND pu.success = 1) as successes,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id) as total_uses
        FROM patterns p
        JOIN pattern_features pf ON p.id = pf.pattern_id
        WHERE {' OR '.join(conditions)}
        GROUP BY p.id
        ORDER BY match_count DESC, successes DESC
        LIMIT ?
    """
    params.append(limit)
    
    return conn.execute(query, params).fetchall()

def record_usage(task_id: str, pattern_name: str, success: bool, notes: str = ""):
    conn = get_db()
    
    # Get pattern id
    pattern = conn.execute("SELECT id FROM patterns WHERE name = ?", (pattern_name,)).fetchone()
    if not pattern:
        print(f"⚠️ Pattern '{pattern_name}' not found")
        return False
    
    conn.execute("""
        INSERT INTO pattern_usage (pattern_id, task_id, success, notes)
        VALUES (?, ?, ?, ?)
    """, (pattern['id'], task_id, 1 if success else 0, notes))
    conn.commit()
    
    status = "✅ success" if success else "❌ failure"
    print(f"Recorded: {pattern_name} on {task_id} → {status}")
    return True

def list_patterns():
    conn = get_db()
    patterns = conn.execute("""
        SELECT p.*,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id AND pu.success = 1) as successes,
               (SELECT COUNT(*) FROM pattern_usage pu WHERE pu.pattern_id = p.id) as total_uses
        FROM patterns p
        ORDER BY p.name
    """).fetchall()
    
    if not patterns:
        print("No patterns stored yet.")
        return
    
    print(f"\n{'Name':<25} {'Success Rate':<15} {'Keywords'}")
    print("-" * 70)
    for p in patterns:
        rate = f"{p['successes']}/{p['total_uses']}" if p['total_uses'] > 0 else "—"
        keywords = (p['keywords'] or "")[:30]
        print(f"{p['name']:<25} {rate:<15} {keywords}")

def show_stats():
    conn = get_db()
    
    total_patterns = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    total_uses = conn.execute("SELECT COUNT(*) FROM pattern_usage").fetchone()[0]
    total_successes = conn.execute("SELECT COUNT(*) FROM pattern_usage WHERE success = 1").fetchone()[0]
    
    print(f"\n=== Pattern Knowledge Base ===")
    print(f"Patterns stored: {total_patterns}")
    print(f"Total uses recorded: {total_uses}")
    print(f"Success rate: {total_successes}/{total_uses} ({100*total_successes/total_uses:.1f}%)" if total_uses > 0 else "No uses yet")
    
    # Top patterns
    top = conn.execute("""
        SELECT p.name, 
               COUNT(*) as uses,
               SUM(pu.success) as wins
        FROM pattern_usage pu
        JOIN patterns p ON pu.pattern_id = p.id
        GROUP BY p.id
        ORDER BY wins DESC
        LIMIT 5
    """).fetchall()
    
    if top:
        print(f"\nTop patterns:")
        for t in top:
            print(f"  {t['name']}: {t['wins']}/{t['uses']} successes")

def get_pattern_details(name: str):
    conn = get_db()
    
    pattern = conn.execute("SELECT * FROM patterns WHERE name = ?", (name,)).fetchone()
    if not pattern:
        return None
    
    features = conn.execute("""
        SELECT feature_type, feature_value FROM pattern_features WHERE pattern_id = ?
    """, (pattern['id'],)).fetchall()
    
    usage = conn.execute("""
        SELECT task_id, success, notes, timestamp FROM pattern_usage 
        WHERE pattern_id = ? ORDER BY timestamp DESC LIMIT 10
    """, (pattern['id'],)).fetchall()
    
    return {
        "pattern": dict(pattern),
        "features": [(f['feature_type'], f['feature_value']) for f in features],
        "recent_usage": [dict(u) for u in usage]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pattern Knowledge Base")
    parser.add_argument("command", choices=["init", "add", "search", "search-features", "list", "stats", "record", "show"])
    parser.add_argument("args", nargs="*")
    parser.add_argument("--description", "-d", default="")
    parser.add_argument("--keywords", "-k", default="")
    parser.add_argument("--code", "-c", default="")
    parser.add_argument("--features", "-f", default="", help="JSON dict of features")
    parser.add_argument("--notes", "-n", default="")
    
    args = parser.parse_args()
    
    if args.command == "init":
        init_db()
    
    elif args.command == "add" and args.args:
        features = json.loads(args.features) if args.features else None
        if features:
            features = list(features.items())
        add_pattern(args.args[0], args.description, args.keywords, args.code, features)
    
    elif args.command == "search" and args.args:
        results = search_patterns(" ".join(args.args))
        if results:
            print(f"\nFound {len(results)} patterns:")
            for r in results:
                rate = f"{r['successes']}/{r['total_uses']}" if r['total_uses'] > 0 else "—"
                print(f"  • {r['name']} ({rate}) — {r['description'][:60]}")
        else:
            print("No matching patterns found.")
    
    elif args.command == "search-features" and args.args:
        features = json.loads(args.args[0])
        results = search_by_features(features)
        if results:
            print(f"\nFound {len(results)} patterns matching features:")
            for r in results:
                print(f"  • {r['name']} (matched {r['match_count']} features)")
        else:
            print("No matching patterns found.")
    
    elif args.command == "list":
        list_patterns()
    
    elif args.command == "stats":
        show_stats()
    
    elif args.command == "record" and len(args.args) >= 3:
        task_id, pattern_name, result = args.args[0], args.args[1], args.args[2]
        record_usage(task_id, pattern_name, result.lower() in ("success", "1", "true", "yes"), args.notes)
    
    elif args.command == "show" and args.args:
        details = get_pattern_details(args.args[0])
        if details:
            print(json.dumps(details, indent=2, default=str))
        else:
            print(f"Pattern '{args.args[0]}' not found")
    
    else:
        parser.print_help()
