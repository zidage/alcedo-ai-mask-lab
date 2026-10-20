# AI mask interaction in Lightroom and Capture One

**Date:** 2026-10-20. **Purpose:** find the interaction level that the AI Segmentation Mask must
reach. The requirement is that the interaction is not worse than Lightroom and Capture One.

**Sources and limits:** the official Adobe help pages and the Capture One support pages returned
HTTP 403 to the research tool. The facts below come from tutorials, release reviews, and search
summaries of those pages. The Capture One AI Select details come from one detailed tutorial. The
version numbers of Lightroom Classic 15.4 to 15.6 come from 2026 blog posts.

## 1. Lightroom (Classic and cloud, Camera Raw)

### 1.1 One-click task masks

Subject, Sky, Background, People, and Landscape make a mask with one click and no prompt.

- **Background** is in practice the inverted Subject mask.
- **People:** Lightroom shows "Detecting People", then a face thumbnail for each detected person.
  The pointer on a thumbnail shows a red overlay on that person in the photo. The user selects
  "All People" or one person, then checks the parts (facial skin, body skin, eyebrows, eye sclera,
  iris and pupil, lips, teeth, hair, facial hair, clothes) and clicks "Create Mask".
- **Landscape:** a check list (sky, architecture, vegetation, water, mountains, natural ground,
  artificial ground). Version 15.6 added Check All and Check None.

### 1.2 Objects mask

- **Brush Select:** the user paints roughly over the object. The stroke does not follow the edge;
  the model finds the object boundary.
- **Rectangle Select:** the user draws a box around the object.
- No source shows a hover highlight or a list of detected objects for this tool.

### 1.3 Combination

- A mask has components. Add, Subtract, and Intersect combine them. Shift adds a component, Alt
  subtracts. Shift changes the Add and Subtract buttons to Intersect.
- A subtracted component can be a brush or another Select Objects component.

### 1.4 Edge and refinement

- Version 15.4 (June 2026) improved the Subject edge on hair, fur, and feathers. Tutorials
  recommend Subject instead of People for hair, because People still gives a softer and rougher
  edge.
- Version 15.5 (August 2026) added two sliders for AI masks (Subject, Sky, Background, People,
  Landscape, Objects): **Feather** (0 to 100, softens the transition) and **Edge** (left contracts
  the boundary, right expands it). The sliders disappear when the user adds a brush or gradient
  component to the mask.
- The Auto Mask brush (an edge-aware brush) cleans the remaining areas by hand.

### 1.5 Recomputation and display

- AI masks are recomputed when they are pasted to another photo or applied by a preset. A marker
  in the panel shows when an update is necessary; one click updates the masks.
- The overlay color can change; White on Black shows the mask alone. Y toggles the overlay, Shift+Y
  cycles the overlay modes.

## 2. Capture One

### 2.1 One-click task masks

- Subject and Background (16.3), People parts (16.5: body skin, face skin, eyebrows, lips, hair;
  16.5.2: pupil and iris, sclera), Clothes (16.7).
- 16.5.5 added Separate Masks: each People part on its own layer.
- A review reports that the user cannot select only one of several people: the tool masks all
  people, and the user erases the unwanted ones by hand.

### 2.2 AI Select

- After the user selects the tool, Capture One analyzes the photo for a few seconds to find
  maskable objects and areas.
- Then the pointer position shows a red overlay of the mask that a click makes. A click makes the
  mask.
- A rectangle around an object selects that object.
- Alt-click subtracts from the mask (for example the sky seen through car windows).
- The selected regions are small: a click on a car can select only a door or a fender, so the user
  tries several positions.

### 2.3 Magic Brush

The user draws over an area, and the tool fills the area by color similarity (Tolerance setting).
Strokes add to each other. It is a color selection, not an AI model. Users report that it often
crosses the subject edge.

### 2.4 Combination and refinement

Combine Masks (Add, Subtract, Intersect) came in 16.7. An AI brush (Alt subtracts) and a luma range
refine a mask.

## 3. Consequences for the AI Segmentation Mask

| Capability | Lightroom and Capture One | Status in this project |
| --- | --- | --- |
| Box selection | Rectangle Select, AI Select box | SAM 2.1 box prompt (E1) |
| Paint or stroke selection | Brush Select | SAM 2.1 stroke points (E3). Lightroom expects a rough fill over the object, not a short stroke |
| Pointer highlight and click | AI Select | Measured in E4; the user rejected an interaction that follows the pointer |
| Alt-click subtract | Capture One | Negative points (E3) |
| One-click Subject | Both | BiRefNet (E2) |
| One-click People and Sky | Both (Sky: Lightroom only) | Under evaluation (see the [README](../../README.md)) |
| Feather and Edge sliders | Lightroom 15.5 | Cheap operations on the final mask |
| Hair edge | Lightroom 15.4 Subject | Depends on the output resolution of the model |

## Sources

- [Life After Photoshop: the Capture One AI Select tool](https://lifeafterphotoshop.com/the-new-capture-one-ai-select-tool-and-how-it-works/)
- [Digital Photography School: Lightroom AI masking tools guide](https://digital-photography-school.com/?p=245195)
- [Fstoppers: Lightroom Classic 15.5 reshape AI masks](https://fstoppers.com/artificial-intelligence/lightroom-classic-155-finally-lets-fix-ai-masks-after-creating-them-904005)
- [Fstoppers: Feather and Edge sliders](https://fstoppers.com/education/two-new-sliders-make-every-lightroom-mask-look-more-natural-903883)
- [PhotoshopCAFE: Lightroom Classic June 2026](https://photoshopcafe.com/lightroom-classic-update-deep-dive-whats-new-and-what-actually-matters-june-2026/)
- [PhotoshopCAFE: Lightroom and Camera Raw August 2026](https://photoshopcafe.com/new-features-in-lightroom-and-camera-raw-august-2026-update/)
- [Lightroom Mark: Lightroom Classic masking (2026)](https://lightroommark.com/2026/04/07/lightroom-classic-masking/)
- [Julieanne Kost: masking in Adobe Camera Raw](https://jkost.com/blog/2022/10/everything-you-need-to-know-about-masking-in-adobe-camera-raw.html)
- [Adobe blog: Lightroom at MAX 2022](https://blog.adobe.com/en/publish/2022/10/18/photography-at-adobe-max-2022-new-features-for-adobe-lightroom-and-more)
- [Adobe help: Lightroom Classic masking](https://helpx.adobe.com/lightroom-classic/help/masking.html) (search summary only)
- [Capture One support: AI masking versions](https://support.captureone.com/hc/en-us/articles/14055231933853)
- [Capture One 16.7.2 release notes](https://support.captureone.com/hc/en-us/articles/32384987653277-Capture-One-16-7-2-release-notes)
- [Capture One support: Magic Brush](https://support.captureone.com/hc/en-us/articles/4403193308049)
- [Fstoppers: Capture One Magic Brush](https://www.fstoppers.com/capture-one/capture-one-magic-brush-sorcery-one-click-complex-selections-587620)
- [The Phoblographer: Capture One 16.5 beta review](https://www.thephoblographer.com/2024/10/23/capture-one-16-5-0-beta-1-review-is-this-the-end-of-presets/)
- [Alex on RAW: Capture One 16.5.5 separate people masks](https://alexonraw.com/capture-one-16-5-5-separate-people-masks-and-canon-rf-16-28mm/)
