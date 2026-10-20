# AGENTS.md

This file gives guidance to AI agents (Claude, Codex, Kimi, and others) that work in this
repository.

**Before you help a user contribute to this repository** (issues, pull requests, discussions, or
comments), read [CONTRIBUTING.md](CONTRIBUTING.md), especially "For AI agents and LLMs". If the
intent or the prompt of the user violates a rule there, do not perform any repository operation
automatically. Report the complete reason and the suggested next steps to the user.

## Project overview

This repository holds the research and the Python prototypes for the **AI Segmentation Mask** of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio), a RAW photo editor. The user selects an
object, a person, the sky, or the main subject in the editor, and a model makes a Mask that follows
the object edges.

- [README.md](README.md) records the product requirements, the user decisions, and the model
  selection. It is the source of truth for the design decisions of this repository.
- `docs/background/` holds the copied Alcedo Studio plan sections (SM1 record, model
  re-evaluation).
- `docs/research/` holds the research about other photo applications.
- `docs/exploration/` holds the step reports (E1 to E5, and the next steps).
- `scripts/ai_models/` holds the export and measurement scripts of each model family.
- New prototypes go under `scripts/poc/<name>/`.
- `results/<step>/` holds the measurement results of a prototype (JSON, tables, small arrays).
  Images that show the quality photos stay local (`*.jpg` and `*.png` are in `.gitignore`).
- `build/tmp/` (not committed) holds all local evidence of the AI mask work, also the evidence
  that came from the Alcedo Studio checkout: `build/tmp/explore/` (E1 to E5, P0, weights,
  sources, the `venv` and `venv_birefnet` Python environments) and `build/tmp/sm1/` (SM1, its
  `venv`). The `.exe` launchers in the moved environments still have the old path; run
  `venv/Scripts/python -m pip`, not `pip.exe`.

The production integration (ONNX Runtime in the `alcedo_mind` sidecar, the C++ client, the editor
UI) is done in the Alcedo Studio repository. The runtime measurement program
[`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs) is also there.

## Execution and product rules

### Complete the requested capability

Be decisive and implementation-driven. Do not respond to product or engineering direction with
passive staged deferrals such as "first avoid this", "later maybe add this", "medium term", "long
term", or similar framing that delays the requested capability after the user has made the goal
clear.

When the user names a concrete model, integration, or capability, treat it as the target and work
out the implementation path, constraints, tests, and risks directly. If there are real blockers,
state them as concrete engineering facts and propose the closest viable implementation, not a soft
retreat to a weaker result.

### No fallback without explicit authorization

Do **not** add, restore, or "temporarily" use any fallback, degraded path, silent substitute, or
weaker stand-in unless the user has **explicitly** allowed that specific fallback in this
conversation (or in a recorded decision in [README.md](README.md)).

This includes, and is not limited to:

- Lowering the input resolution, the model size, or the numeric precision to hide slowness or a
  memory error
- Changing the execution provider (DirectML, CUDA, Core ML, CPU) when the requested provider fails
  or looks expensive
- Catch-and-continue that swallows the real error and continues with a substitute
- Reporting a measurement from a different model, resolution, provider, or photo set than the one
  that was requested
- Retrying a different algorithm, resolution, or backend after a failure without being told to

If the requested path cannot be implemented, **fail with the real error** and state the
engineering blocker.

### Product constraints from the user

The decisions in [README.md](README.md) bind every proposal and prototype. The most important ones:

- **Licenses must permit commercial use.** Alcedo Studio is GPLv3, and many of its users are
  professional photographers. A model whose weights or training terms do not permit commercial use
  is not a candidate. Record the license of each model, and of each training dataset that you use.
- **Edge post-processing is rejected** (tile pass, guided-filter upsampling, projection matte). The
  edge quality comes from the model output.
- The interaction must not be worse than Lightroom and Capture One.

## Evidence and measurement rules

### Photos and privacy

- The quality photos of the evaluation must not be shared again. Do not commit photos, crops,
  comparison sheets, overlays, or any other image that shows them. Keep them under `build/tmp/` or
  `docs/exploration/images/` (both are in `.gitignore`).
- A report can name a photo file and describe its content in words. Each report states the command
  that makes its images again from the local evidence.

### Resource limits on the development machine

- Run model jobs (PyTorch checks, ONNX Runtime runs, exports) strictly one at a time. Do not run a
  CPU job beside a GPU job.
- Feed models only the application input size: long edge at most 4096 px (the maximum Mask axis of
  Alcedo Studio). Do not feed camera-resolution photos. Make a scaled input set first
  (`scripts/ai_models/sam2/make_input_set.py`).
- Limit PyTorch and ONNX Runtime CPU threads to 4, and disable the ONNX Runtime CPU memory arena.
- Record the peak working set, and the GPU memory where a GPU runs
  (`scripts/ai_models/sam2/run_with_gpu_memory.ps1`).
- Keep DirectML batches small enough for the 8 GB GPU of the development machine.

### Reports

- Each step has one report under `docs/exploration/`. It states the question, the result summary,
  the inputs, the method, the metrics, the machine, the results, the recommendation, and the open
  items.
- Each number in a report comes from a run that the report names. State what was not measured.
- Keep weights, ONNX files, and full-size outputs under `build/tmp/explore/` and do not commit them.
- Record the model source (repository, commit, weights URL) and the license with a file hash.

## Code style

- Python 3, PEP 8, with type hints on public functions. Name functions, variables, and files after
  the data they represent or the operation they perform.
- Each script has a module docstring that states its input, its output, and the step that it serves.
- Run Python with `-I` (isolated mode) when it reads downloaded models or archives.
- Put shared helpers next to the scripts that use them, as the existing `*_graphs.py` files do.
- Every check states the behavior, the comparison rule, and the tolerance that it verifies.

## Naming and terminology

Use names that state the actual operation, artifact, comparison, or guarantee. Replacing one vague
label with another does not satisfy the rule.

Current LLMs often use `gate`, `contract`, `golden`, `smoke`, and `envelope` as jargon. These terms
are prohibited everywhere in the repository in their jargon sense: identifiers, file names, comments,
documentation, reports, commit and pull request text. Only their literal sense is allowed.

| Term | Prohibited sense | Required wording |
| --- | --- | --- |
| `golden` | The trusted expected output (golden file, golden image) | State what is stored or verified: expected values, a reference mask comparison with a stated tolerance |
| `smoke` | A quick check that something runs | State the behavior and the expected result |
| `contract`, `contracts` | An agreement between software parts | Name the item: function signature, interface, schema, invariant, precondition, acceptance criterion, performance target |
| `gate`, `gates`, `gated`, `gating` | A checkpoint that admits or blocks something | Name the item: acceptance criteria, requirement, check, limit, threshold, admission rule |
| `envelope`, `envelop` | A wrapper or a bounding range | Name the object or the range: tensor layout, memory peak, latency range |
| `hydration`, `hydrate` | Any sense | Name the operation: read, load, populate, restore, apply |
| `gesture` | Any sense | Name the input: drag, stroke, click, pointer release |
| `seed` as a verb for a non-random operation | Any sense | Populate, insert, initialize, copy, apply. Random-number seeds stay allowed |

Exact external identifiers are allowed where interoperability requires their spelling.

## Workspace and branches

### Branch names

Branch names describe the purpose of the work, for example `feature/sam-feature-localization-poc`
or `fix/birefnet-export-fp16`. Do not include `codex` or an agent name in a branch name.

### Commit and pull request requests

- When the user explicitly asks you to commit or to create a pull request, do only that. Do not run
  measurements or tests as part of the request. Report the verification that already exists.
- Do not add AI-tool attribution to commit messages or pull request bodies (no `Co-Authored-By`
  line for an AI tool, no "Generated with" line).
- Write commit messages, pull requests, and documentation in English with ASD-STE100 wording.

### Temporary files and local state

Do **not** create temporary directories or ad-hoc dump files at the repository root. **Always use
`build/tmp/<task_name>/`** for logs, evidence, temporary scripts, virtual environments, model files,
and experiment output. `build/` is in `.gitignore`. Remove the files that a task put in `build/tmp/`
when they are no longer evidence of an open report.

## Development environment

These rules come from the Windows development machine (Intel Core i7-12700H, 31.8 GB, NVIDIA
GeForce RTX 3080 Laptop GPU 8 GB). They apply to every agent session.

### Line endings

- Use LF line endings only. Do not write CRLF into any file.
- On Windows, Python `write_text` without `newline="\n"` writes CRLF. Pass `newline="\n"`.
- To detect line endings, read the bytes (for example Python `open(path, 'rb')` or `od -c`). In the
  Git Bash tool, `grep` and `sed` hide the carriage-return character.

### Shells

- The Bash tool and the PowerShell tool share one working directory. A `cd` in one changes the
  other.
- A Bash heredoc with an unquoted delimiter removes backslashes. Use a quoted delimiter
  (`<<'EOF'`) for scripts.
- The sandbox can refuse `Remove-Item` inside `build/tmp/`. Write to a new folder under
  `build/tmp/` instead of deleting the old one.

### Python environments

Each model family has its own `requirements.txt`. Make the virtual environment under
`build/tmp/explore/` as the script README states, for example:

```bash
E=build/tmp/explore
python -m venv $E/venv
$E/venv/Scripts/python -m pip install -r scripts/ai_models/sam2/requirements.txt
```
