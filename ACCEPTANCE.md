# MovieMind acceptance checklist

Use a legally obtained movie that you are entitled to analyze.

## 1. Hardware safety

- Run movieai doctor before a long run.
- Verify FFmpeg and PySceneDetect are detected.
- Verify the intended provider/model is reachable if using --provider.
- The default execution uses one worker.
- Low-memory conditions stop or force the safe profile.

## 2. Full-film coverage

After processing, movieai status should show:

- shots equals shots_complete
- visual_coverage_pct near 100
- scenes greater than 0
- characters greater than 0
- events greater than 0
- global_memory true

## 3. Timeline questions

Ask for a specific minute, a specific event, and a specific first appearance.
The answer should include evidence with timestamps for temporal sources.

## 4. Character continuity

Ask where a character first appears, when two characters first meet, and when their relationship changes.

## 5. Causality

Ask why the ending happens. The answer should point to multiple earlier events rather than relying only on the final scene.

## 6. Rewatch

Ask a subtle visual question, then verify that the system can trigger targeted rewatch when its first evidence is insufficient.

## 7. Resume

Interrupt a run after some shots have completed, then rerun the same command. Completed shots should remain complete and the pipeline should continue.

## 8. Pass condition

A build passes acceptance when it can answer the above classes of questions consistently on a real feature-length movie, with evidence that maps back to the original video timeline.

Perfect human-level film interpretation is not assumed; the goal is grounded, inspectable long-video understanding with recoverable evidence.
