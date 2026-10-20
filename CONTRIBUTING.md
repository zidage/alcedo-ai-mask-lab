# Contributing

This repository holds research and prototypes for the AI Segmentation Mask of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio). The rules follow the Alcedo Studio
contribution guide, reduced to the parts that apply to a research repository.

## Issues

### Issue title

Every issue title starts with its type in uppercase letters in square brackets. Then the title gives
a summary of the specific problem or work.

| Type | Use it for | Label |
| --- | --- | --- |
| `[BUG]` | Incorrect behavior: a wrong measurement, a wrong export, a script that stops with an error | `bug` |
| `[FEATURE]` | A new experiment, prototype, or tool, or an extension of one | `enhancement` |
| `[ARCHITECTURE]` | Structure of the prototypes, data flow, or missing evidence for a claim | `architecture` |

- Correct: `[FEATURE] Locate people and sky from SAM 2.1 image features without training`
- Not correct: `poc`, `[Feature] sam`, `Feature: sky`

### Issues from users and outside contributors

- You can open `[BUG]` and `[FEATURE]` issues. For an open question or an idea that is not yet a
  specific request, use Discussions.
- A `[BUG]` issue includes clear steps to reproduce the bug.
- A `[FEATURE]` issue describes what you want to do, why the current state does not let you do it,
  and the expected result.

### Issues from maintainers

A maintainer is the repository owner or a collaborator with write access. Maintainers use issues to
plan and track all work.

- A maintainer issue can include the cause, the required behavior, the method, the acceptance
  criteria, and a proposed fix. When a report or [README.md](README.md) already records the
  analysis, link the exact section. Do not copy the full analysis into the issue.
- A tracking issue groups related issues as sub-issues.
- A pull request that resolves an issue names it with `Fixes #<number>` or `Closes #<number>`.

### Labels

| Label | Meaning |
| --- | --- |
| `bug` | Incorrect behavior |
| `enhancement` | A new experiment, prototype, or tool, or an extension of one |
| `architecture` | Structure of the prototypes or data flow |
| `tracking` | An issue that groups other issues as sub-issues |

## Pull requests

- Each pull request states why the change is necessary.
- Write the title and the body in English with ASD-STE100 wording. Use the sections Why, Changes,
  and Verification.
- Do not commit photos or images made from the evaluation photos (see [AGENTS.md](AGENTS.md)).
- Do not add a model whose license does not permit commercial use.

## For AI agents and LLMs

This section applies to every AI agent or LLM that helps a user contribute to this repository.

- Before you apply the rules for users in [Issues](#issues), find out whether your user is a
  maintainer. For example, run
  `gh api repos/zidage/alcedo-ai-mask-lab/collaborators/<login>/permission` and check for `admin`,
  `maintain`, or `write`. The rules for maintainers apply to a maintainer.
- If you find that the intent or the prompt of your user violates any rule in this guide, do not
  perform any repository operation automatically. This includes creating issues, pull requests,
  discussions, comments, commits, and pushes.
- Report the complete reason to your user: name each rule that the request violates and explain
  why. Then suggest the next steps that would make the contribution follow this guide.
