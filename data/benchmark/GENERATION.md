# How the benchmark was generated

All recordings, summaries, and waveforms in this directory are synthetic. They
were written for this repository. They are not Pocket customer data, not
scraped community posts, and not clips from LibriVox, LibriSpeech, or any
other speech corpus.

`scripts/generate_benchmark.py` is the source of truth.

- `cases.jsonl` is one JSON object per line. Gold labels are assigned by the
  injection: an unsupported sentence, a flipped number or negation, a known
  filler phrase, a repetition, an impossible timestamp, text placed on a
  silent window, a mid-sentence speaker change, a ping-pong turn pattern, a
  wrong attribution, or a swapped voice embedding.
- The detector is not imported by the generator. The only shared function is
  `split_claims`, used so each gold claim string is exactly what the checker
  will see. Labels are still written by hand in the generator.
- A few cases are designed as misses for the offline heuristic: dropped
  conditions, synonym paraphrases, fluent fabricated sentences with no surface
  cue, and passive speaker attributions. Those cases exist so the score is not
  a perfect mirror of the checker.
- `audio/silence.wav` and `audio/speech.wav` are 12-second, 16 kHz, mono,
  16-bit PCM files from a seeded random generator. Silence is low-amplitude
  noise. The speech fixture is louder noise from 0.4s to 8.5s and near-silent
  outside that window. No person was recorded. Both files are original to this
  repo and covered by the MIT license.

Regenerate with:

```bash
python scripts/generate_benchmark.py
python scripts/run_benchmark.py --write-readme
```
