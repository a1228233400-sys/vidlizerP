"""Human-readable MovieMind report generation."""

from __future__ import annotations

from pathlib import Path

from .db import connect, get_global_memory


def _fmt(seconds: float) -> str:
    total = max(0, int(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_report(db_path: Path) -> str:
    conn = connect(db_path)
    try:
        movie = conn.execute("SELECT * FROM movies ORDER BY id DESC LIMIT 1").fetchone()
        if not movie:
            raise RuntimeError("No movie found in database.")
        mid = int(movie["id"])
        shots = int(conn.execute("SELECT COUNT(*) FROM shots WHERE movie_id=?", (mid,)).fetchone()[0])
        complete = int(conn.execute(
            "SELECT COUNT(*) FROM shots WHERE movie_id=? AND analysis_status='complete'", (mid,)
        ).fetchone()[0])
        scenes = int(conn.execute("SELECT COUNT(*) FROM scenes WHERE movie_id=?", (mid,)).fetchone()[0])
        chars = int(conn.execute("SELECT COUNT(*) FROM characters WHERE movie_id=?", (mid,)).fetchone()[0])
        events = int(conn.execute("SELECT COUNT(*) FROM events WHERE movie_id=?", (mid,)).fetchone()[0])
        transcript = int(conn.execute("SELECT COUNT(*) FROM transcripts WHERE movie_id=?", (mid,)).fetchone()[0])
        evidence = int(conn.execute("SELECT COUNT(*) FROM evidence WHERE movie_id=?", (mid,)).fetchone()[0])
        duration = float(movie["duration_s"])
        covered = float(conn.execute(
            "SELECT COALESCE(SUM(duration_s),0) FROM shots WHERE movie_id=? AND analysis_status='complete'",
            (mid,),
        ).fetchone()[0])

        memory = get_global_memory(conn, mid) or {}
        coverage = covered / duration * 100 if duration > 0 else 0.0

        lines = [
            "# MovieMind Report",
            "",
            f"Source: {movie['source_path']}",
            f"Duration: {_fmt(duration)}",
            "",
            "## Coverage",
            f"- Shots detected: {shots}",
            f"- Shots analyzed: {complete}",
            f"- Visual timeline coverage: {coverage:.2f}%",
            f"- Transcript segments: {transcript}",
            f"- Evidence records: {evidence}",
            "",
            "## Story Memory",
            f"- Scenes: {scenes}",
            f"- Characters: {chars}",
            f"- Events: {events}",
            "",
        ]

        if memory:
            if memory.get("logline"):
                lines += ["## Logline", str(memory["logline"]), ""]
            if memory.get("full_summary"):
                lines += ["## Full Summary", str(memory["full_summary"]), ""]
            acts = memory.get("acts") or []
            if acts:
                lines.append("## Acts")
                for act in acts:
                    lines += [
                        f"### {act.get('title', 'Act')}",
                        str(act.get("summary", "")),
                        "",
                    ]
            arcs = memory.get("character_arcs") or []
            if arcs:
                lines.append("## Character Arcs")
                for arc in arcs:
                    lines.append(
                        f"- {arc.get('character_key', 'unknown')}: {arc.get('arc', '')}"
                    )
                lines.append("")
            themes = memory.get("major_themes") or []
            if themes:
                lines += ["## Themes", *[f"- {theme}" for theme in themes], ""]
            foreshadowing = memory.get("foreshadowing_candidates") or []
            if foreshadowing:
                lines.append("## Foreshadowing Candidates")
                for item in foreshadowing:
                    lines.append(
                        f"- Setup: {item.get('setup', '')} | Payoff: {item.get('payoff', '')}"
                    )
                lines.append("")

        return "\n".join(lines).rstrip() + "\n"
    finally:
        conn.close()


def write_report(db_path: Path, output: Path | None = None) -> Path:
    output = output or db_path.with_suffix(".report.md")
    output.write_text(render_report(db_path), encoding="utf-8")
    return output
