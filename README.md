# Alcedo AI Mask Lab

Research and Python prototypes for the **AI Segmentation Mask** of
[Alcedo Studio](https://github.com/zidage/AlcedoStudio), a RAW photo editor. In the editor, the user
selects an object, a person, the sky, or the main subject, and a model makes a Mask that follows the
object edges. The Mask joins the existing Mask system of a Color Grade node.

This README records the product requirements, the user decisions, and the current model selection.
The evidence is in the step reports under [`docs/exploration/`](docs/exploration/README.md).

**Status (2026-10-20):** the exploration steps E1 to E5 are done. The model set is under decision.
The next step is the proof of concept "locate people and sky from the SAM 2.1 image features"
(section 5).

## Main project

| Item | Link |
| --- | --- |
| Repository | [zidage/AlcedoStudio](https://github.com/zidage/AlcedoStudio) |
| Re-evaluation issue | [AlcedoStudio#458](https://github.com/zidage/AlcedoStudio/issues/458) |
| Master plan | [AI Segmentation Mask master plan](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/docs/roadmap/alcedo_studio/edit/ai_segmentation_mask_master_plan.md) (decisions D1 to D11 in section 1.2) |
| Branch of the research | [`feature/ai-segmentation-sm1-quality-review`](https://github.com/zidage/AlcedoStudio/tree/feature/ai-segmentation-sm1-quality-review) |
| Runtime measurement program | [`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs) |

The production integration (sidecar inference, C++ client, editor UI) is done in Alcedo Studio.
This repository decides the models and the interaction, and proves them in Python.

## 1. Background

The first plan used EfficientSAM3 for point, box, and text prompts. Its quality review failed: the
point and box path makes regular grid holes on large regions and fails on close-up portraits,
because the student image encoder gives a 32 x 32 feature map that is resized to the 72 x 72 grid of
SAM3. Single points are ambiguous, and the text path misses many common classes. Details:
[model re-evaluation](docs/background/model_re_evaluation.md) and
[SM1 record](docs/background/sm1_efficientsam3_record.md).

The re-evaluation (Alcedo Studio issue
[#458](https://github.com/zidage/AlcedoStudio/issues/458)) measured these candidates:

| Step | Report | Result |
| --- | --- | --- |
| E1 | [SAM 2.1 point and box prompts](docs/exploration/e1_sam2_1_point_box_evaluation.md) | No grid holes in any size; the close-up box selects the whole person. Tiny: 76 ms encoder, 10.5 ms decoder, 1.36 GB on DirectML |
| E2 | [BiRefNet one-click Subject](docs/exploration/e2_birefnet_subject_matting.md) | Lite finds the expected subject on 12 of 13 photos without a prompt, with a clean but hard hair edge. Matting variants miss rigid objects |
| E3 | [Stroke selection](docs/exploration/e3_stroke_selection.md) | A stroke across the region with 3 or 4 points selects the intended region without a level change on 23 of 25 targets |
| E4 | [Click selection with candidates](docs/exploration/e4_click_candidates.md) | Precomputed candidates are slower and not better than the live point decoder. The user rejected an interaction that follows the pointer |
| E5 | [Edge stage](docs/exploration/e5_edge_stage.md) | A tile pass with guided-filter upsampling improves the edge but costs 0.15 to 1.5 s. The user rejected this edge post-processing |

The interaction of other applications is in
[AI mask interaction in Lightroom and Capture One](docs/research/interaction_practice.md).

## 2. User decisions

These decisions come from the user. Do not change them without a new user decision.

| ID | Decision | Date |
| --- | --- | --- |
| U1 | The interaction must not be worse than Lightroom and Capture One. | 2026-10-20 |
| U2 | Model licenses must permit commercial use. Alcedo Studio is GPLv3 and the user does not sell it, but many users are professional photographers whose use is commercial. Non-commercial weights are excluded. AGPL-3.0, Apache-2.0, MIT, and the Meta SAM License are acceptable. | 2026-10-20 |
| U3 | Edge post-processing is rejected: the tile pass, guided-filter upsampling, and the projection matte are too slow and give poor results. The edge quality comes from the model output. | 2026-10-20 |
| U4 | One-click People and one-click Sky are required. | 2026-10-20 |
| U5 | The selection of one person is more important than recall in crowds. | 2026-10-20 |
| U6 | GPU memory: the editor uses about 5 GB on a 24 MP photo. The 3 GB budget for the models is soft, because the driver can swap part of the editor memory (caches). Models run while the user selects; they are unloaded when the user goes back to adjusting the photo. | 2026-10-20 |
| U7 | An interaction that follows the pointer (hover highlight) is rejected. The stroke is the prompt tool (E3). | 2026-10-20 |
| U8 | Float16 models are acceptable (lower precision for a shorter processing time). | 2026-10-19 |
| U9 | If BiRefNet is used, use the 512 px variant; efficiency first. See open decision O2. | 2026-10-20 |
| U10 | A comparison experiment of prompt sources for People and Sky ("E7") is not done. | 2026-10-20 |
| U11 | The SAM3 detector is too large for its task (locating people and sky). Evaluate a small model that gives points or a box to SAM 2.1; a model trained by the user (for example on Colab) is acceptable. | 2026-10-20 |

## 3. Product requirements (proposal)

The requirement set is a proposal. The user has not approved each row.

| ID | Function | Reference | Proposed priority | Acceptance |
| --- | --- | --- | --- | --- |
| F1 | Select Subject (one click) | Lightroom, Capture One | Required | No prompt. All salient subjects in one mask. A photo without a subject gives a "not found" message, not an empty mask |
| F2 | Select Background | Lightroom, Capture One | Required | The inverted F1 mask; no extra model |
| F3 | Select Sky (one click) | Lightroom | Required (U4) | No blocks or glow along tree or building horizons. A photo without sky gives a "not found" message |
| F4a | Select People (one click): all people or one person | Lightroom People list | Required (U4, U5) | A list of the detected people; the user can select one person. Missed people in crowds are acceptable |
| F4b | People parts (skin, hair, clothes, eyes) | Lightroom, Capture One | Not in this release | Needs a human parsing model |
| F5 | Object selection: stroke and box | Lightroom Objects, Capture One AI Select box | Required | A click is a stroke with one point. No pointer-follow highlight (U7) |
| F6 | Add and subtract within one AI Mask | Lightroom, Capture One | Required | Strokes and boxes as model prompts; Alt subtracts (E3) |
| F7 | Combination across mask types (Subject minus a stroke, People intersect Sky) | Lightroom, Capture One 16.7 | Required; needs a user decision | Alcedo Studio decision D6 has only Union (per-pixel maximum). Subtract and Intersect need a composition of the components into one R8 coverage at commit |
| F8 | Feather and Edge sliders | Lightroom 15.5 | Required | Blur and erode or dilate of the final mask; adjustable after commit. Unlike Lightroom, they stay available when a brush component is added |
| F9 | Store the recipe (task and prompts) and recompute | Lightroom paste and presets | Needs a user decision | Store the recipe at least; recompute on paste or sync is an extension |
| F10 | Overlay: color, opacity, toggle key | Lightroom Y key | Required | Alcedo Studio decision D3 |
| F11 | Text prompts | Neither application | Proposed for removal | Removes the text requirement of Alcedo Studio decision D2 |

## 4. Model selection (proposal)

Principle: **SAM 2.1 is the only interactive model.** A task model either makes its own mask
(Subject) or gives prompts (points or a box) to SAM 2.1 (People, Sky).

| Function | Model | License | Download | Evidence and open items |
| --- | --- | --- | --- | --- |
| F5, F6 (and the final mask of F3, F4a) | SAM 2.1 Hiera tiny | Apache-2.0 | 130.6 MB (encoder 109.4 MB, decoder 21.2 MB, float32) | E1, E3. Mask logits are 256 x 256: one logit cell is 16 px at the 4096 px application input |
| F1, F2 | BiRefNet (variant: open decision O2) | MIT | lite float16: 96.4 MB; `BiRefNet_512x512` (Swin-L): 444 MB | E2 measured lite at 1024 px (float16: 175 ms, 3.2 GB). The 512 px variant is not measured |
| F3, F4a localization | A small head on the SAM 2.1 image features (section 5) | Own model; the training data license must permit commercial use | Less than 1 MB (estimate) | Proof of concept open. Person alternative without training: YOLOX-Nano (Apache-2.0) |

Estimated total download: about 228 MB with BiRefNet lite and the small head.

### 4.1 How People and Sky work

```text
Open the AI Mask tool: SAM 2.1 encoder (once per photo) -> image features
Select Sky:    head reads the features -> sky score map -> a few confident points -> SAM 2.1 decoder -> mask
Select People: head reads the features -> person score map -> one region per person
               -> box or points per person -> SAM 2.1 decoder -> one mask per person -> People list
No sky or no person: the highest score is below the threshold -> "not found" message
```

The SAM 2.1 encoder already reads the whole photo for the interaction. The head only reads the
features, so People and Sky add no second image encoder.

### 4.2 Candidates that were replaced or rejected

| Candidate | Reason |
| --- | --- |
| EfficientSAM3 point and box path | Grid holes, close-up failures (SM1, section 1) |
| EfficientSAM3 text detector as the People and Sky localizer | 201.6 MB (plus a 169.2 MB text encoder), 254 ms, 2.4 s session load, Core ML profile did not finish; misses people in crowds. Replaced by the proposal of U11 |
| BiRefNet general, HR, and matting variants | Speed and memory (E2), matting variants miss rigid objects; user decision 2026-10-19 |
| SAM 2.1 automatic mask generator candidates | 0.8 to 3.2 s and 1.3 GB more GPU memory, not better than the live decoder (E4) |
| Tile pass, guided filter, projection matte | U3 |
| SegFormer (ADE20K) for Sky | NVIDIA non-commercial license (U2) |
| Sapiens for People parts | Non-commercial license (U2) |
| ViTMatte and other trimap matting models | Trained on Composition-1k (research use only); also an edge post-processing stage (U3) |
| HQ-SAM 2 | Only the Hiera large size (3.1 GB, 322 ms encoder in E1). Kept as an option for better object edges |

### 4.3 SAM 2.1 derivatives to check

| Model | License | Reason to check |
| --- | --- | --- |
| UnSAMv2 | Apache-2.0 | Granularity control for single-point ambiguity (part or whole object). The backbone size is not confirmed |
| EdgeTAM | Apache-2.0 | RepViT encoder of 8.3M parameters (33 MB). Only if tiny is too slow on a target machine. Check the grid-hole probe of E1 first |
| EfficientTAM | Apache-2.0 | Not preferred: plain single-scale ViT, the same cause as the EfficientSAM3 grid holes is possible |

## 5. Next step: locate people and sky from SAM 2.1 image features

The SAM3 detector is open-vocabulary: it can find any word. The product uses two fixed words, so it
pays for generality that it does not use. People and Sky need only a coarse answer: "is it there,
and where are a few reliable points". The SAM 2.1 encoder already computes a 64 x 64 feature map of
the whole photo.

The proof of concept checks, without training, whether these features separate people and sky
from the rest of the photo. If they do, a small head (one to three convolution layers) is trained on
the frozen SAM 2.1 tiny encoder. If they do not, the person localizer changes to a small detector,
and the sky needs a different feature source.

Risks:

- SAM features carry more shape and edge information than class information. Linear probes of the
  SAM encoder for classification are clearly worse than CLIP or DINOv2.
- Two people next to each other give one region in a score map. A center-point output can separate
  them; U5 makes this less important.
- Branches cut the sky into pieces. Each confident piece needs its own points.
- Training data: COCO-Stuff (CC BY 4.0 annotations, Flickr images with mixed licenses) or
  Open Images V7 (CC BY 4.0 annotations, CC BY 2.0 images). The license of each dataset must be
  checked against U2.

## 6. Repository layout

| Path | Content |
| --- | --- |
| [`docs/background/`](docs/background/) | Copied Alcedo Studio plan sections: SM1 record, model re-evaluation |
| [`docs/research/`](docs/research/) | Interaction research of other applications |
| [`docs/exploration/`](docs/exploration/README.md) | Step reports E1 to E5 (images local only) |
| [`scripts/ai_models/sam2/`](scripts/ai_models/sam2/README.md) | SAM 2.1 export, probes, stroke and candidate tools |
| [`scripts/ai_models/birefnet/`](scripts/ai_models/birefnet/README.md) | BiRefNet export, float16 conversion, matte probes |
| [`scripts/ai_models/efficientsam3/`](scripts/ai_models/efficientsam3/README.md) | EfficientSAM3 export and comparison (SM1) |

The scripts came from Alcedo Studio commit `d1e0d8c7` (branch
`feature/ai-segmentation-sm1-quality-review`). Issue numbers in the reports (for example [AlcedoStudio#458](https://github.com/zidage/AlcedoStudio/issues/458))
refer to the Alcedo Studio repository. The runtime measurements use
[`rust/puerh_mind/examples/segmentation_runtime_report.rs`](https://github.com/zidage/AlcedoStudio/blob/feature/ai-segmentation-sm1-quality-review/rust/puerh_mind/examples/segmentation_runtime_report.rs) in Alcedo Studio.

Rules for contributors and agents: [AGENTS.md](AGENTS.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## 7. Open decisions

| ID | Question |
| --- | --- |
| O1 | Approve the requirement set of section 3, especially F7 (combination across mask types) and F9 (recipe and recompute). |
| O2 | BiRefNet variant. U9 asks for the 512 px variant. Because U3 removed edge post-processing, the output resolution is the final edge resolution: lite at 1024 px gives 4 px cells at the 4096 px input (measured: 175 ms, 96.4 MB) and `BiRefNet_512x512` gives 8 px cells (Swin-L, 444 MB, not measured). The recommendation is lite at 1024 px. |
| O3 | Remove the text prompt requirement from Alcedo Studio decision D2 (F11). |
| O4 | Person localizer: the SAM 2.1 feature head, or YOLOX-Nano, after the proof of concept. |

## License

GPL-3.0, the same as Alcedo Studio. See [LICENSE](LICENSE). Model weights are not part of this
repository; each model keeps its own license.
