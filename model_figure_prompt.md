# Task: draw the model architecture figure in PowerPoint (Mac) for a paper

You are going to draw one publication-quality architecture figure in Microsoft PowerPoint on a Mac. The figure goes into a research paper, so it must be accurate to the model described below, readable when printed small, and built from editable native shapes (no pasted screenshots).

You do not have access to the code. Everything you need about the model is in this brief. Do not add, rename or guess components that are not listed here. Where something is marked "confirm with the author", ask before finalizing instead of inventing an answer.

## What the model does

It classifies which of five on-screen options (A to E) a person chose on a phone, from a short front-camera video clip. Two feature streams are extracted from the same clip, each as a time series, and a small Transformer classifies the pair.

## Architecture facts (exact, as currently configured)

Notation: B = batch, T = 50 time steps, d = 16 (model width), M = 2 modalities, C = 5 classes.

**Inputs (one clip)**

| Stream | Name to show in the figure | Shape per clip | What it is |
|---|---|---|---|
| 1 | Eye feature | T × 39 | per-frame eye/gaze/face feature vector from the front camera |
| 2 | Optical flow | T × 8 | RAFT optical flow averaged over a 2 × 2 grid of the frame; (u, v) per cell |

Both streams come from the same video, are stored at 80 frames and resampled to T = 50 before the model. Show this only as a small "resample to T = 50" note, not as a block of the network.

**Per-modality temporal encoder (two separate copies, no weight sharing)**

1. Linear projection: input dim → 16, multiplied by √16.
2. Fixed sinusoidal positional encoding (added), with dropout.
3. Transformer encoder, 3 layers. Each layer:
   - multi-head self-attention over time (4 heads), then add & BatchNorm
   - feed-forward: Linear 16 → 256, GELU, Linear 256 → 16, then add & BatchNorm
4. GELU, then dropout (0.2).

Output of each encoder: a sequence of T tokens of size 16.

Note that the normalization inside the encoder is **BatchNorm**, not LayerNorm. Label it that way.

**Same-time cross-modal fusion**

- At each time step t, the M = 2 tokens (eye token at t, optical-flow token at t) form a tiny sequence of length 2.
- One multi-head attention layer (4 heads) runs over those 2 tokens, followed by add & LayerNorm.
- Attention is only between modalities at the same time step. There is no attention across time steps in this block.
- Output: for each modality, T fused tokens of size 16.

This is the key idea of the figure. Draw it as T narrow columns (show t = 1, 2, 3, …, T), each column holding one eye token and one optical-flow token joined by a double-headed arrow.

**Classifier head**

1. Apply the padding mask, concatenate the two modalities per time step → T × 32.
2. Flatten → 1600.
3. Linear 1600 → 16, GELU, Dropout (0.2), Linear 16 → 5.
4. Output: class logits for A to E.

**Auxiliary heads (training only)**

- One per modality, taking that modality's tokens **after** the fusion block.
- Masked mean pooling over time → Linear 16 → 16, GELU, Dropout, Linear 16 → 5.
- Not used at test time. Draw with a dashed outline and the label "training only".

**Training loss**

L = CE(fused logits) + λ · mean over modalities of CE(auxiliary logits), with λ = 0.1.

**Sizes worth printing in a caption or small note**

d = 16, 4 heads, 3 encoder layers, feed-forward width 256, dropout 0.2, T = 50, 5 classes, about 86k parameters in total (about 29k per encoder).

## Layout

Left-to-right flow, single wide panel:

1. Far left: the two input streams, stacked vertically, each drawn as a strip of thin frame cells with its name and shape.
2. Two parallel temporal-encoder blocks, one per stream, each showing: Linear projection → positional encoding → Transformer encoder × 3. An optional small inset may expand one encoder layer (self-attention, add & BatchNorm, feed-forward, add & BatchNorm).
3. Centre: the same-time cross-modal fusion panel described above. Make this the visual focus.
4. Right: mask · concat · flatten → classifier head → fused logits (5 classes).
5. Below the fusion panel: the dashed auxiliary branch (mean pool → auxiliary head, × M), with an arrow into a loss box that also receives the fused logits.

Keep arrows horizontal or vertical only (elbow connectors, no diagonals), and do not let any arrow cross a label.

## Visual style

- **Palette: Morandi (muted, grey-toned).** Use one hue per role and keep it consistent:
  - eye stream: dusty blue (fill `B4C3CE`, outline `7F95A6`)
  - optical-flow stream: dusty rose (fill `DCC1BD`, outline `B08E8A`)
  - fusion: sage green (fill `C3CFBE`, outline `8FA28A`, light panel `E8EDE5`)
  - classifier head: mauve (fill `CFC5D3`, outline `9A8BA0`)
  - auxiliary branch: sand (fill `E3D7BE`, outline `B5A27C`), dashed outline
  - loss: taupe (fill `C9C1B9`, outline `8A8179`)
  - text `45464A`, arrows `77787B`, white background
- Rounded rectangles with thin (0.75 to 1 pt) outlines, no shadows, no gradients, no 3D effects.
- One sans-serif font throughout (Arial or Helvetica). No text smaller than 8 pt at final print size.
- No slide title, no decorative elements, no logo. The paper supplies the caption.
- Colours must not be the only cue: every block carries a text label.

## Size and export

- Set the slide to the figure's real size. Default to a full-width two-column figure: **7.0 in wide by about 3.2 in tall**. Confirm the target width with the author if you can; if the paper is single-column, use 3.4 in wide and simplify labels.
- Group related shapes (each encoder, the fusion panel, the head, the auxiliary branch) so the author can move them.
- Deliver two files: the editable `.pptx`, and a vector **PDF** exported from it (File → Export → PDF). Check that the PDF text is selectable, which shows it stayed vector.

## Checks before you hand it over

- Every number in the figure matches this brief (39, 8, 16, 256, 1600, 5, T = 50, 3 layers, 4 heads, λ = 0.1).
- The encoder shows BatchNorm; the fusion block shows LayerNorm.
- The fusion arrows connect the two modalities within a time step only, never across time steps.
- The auxiliary branch starts after the fusion block, is dashed, and is marked "training only".
- At 100% zoom at the final size, all text is readable and nothing overlaps or is clipped.
- View the exported PDF, not just the slide, before reporting it done.

## Confirm with the author

1. The display names of the two streams ("Eye feature" and "Optical flow" are the working names).
2. The target figure width (two-column or single-column).
3. Whether the paper's final model is the same-time fusion above or the **gated variant** below. Draw the same-time version unless told otherwise.

## Optional variant: gated fusion with a frozen eye branch

Only draw this if the author asks for it. It is a two-stage training scheme built on the same parts:

- Stage 1: an eye-only model is trained (eye encoder → single-modality fusion block → classifier).
- Stage 2: that eye path is copied in and frozen (mark it with a lock or "frozen" tag). The two-modality path above (both encoders → same-time fusion → classifier) produces a correction.
- Output: logits = eye-path logits + g · correction logits, where g is a single learned gate initialized at 0, so the model starts out identical to the eye-only model.

In the figure this adds a second, frozen classifier path from the eye encoder, a multiplication by g on the fused path, and a sum node before the output.

## What to report back

A short note with: the two file paths, the final figure size, anything you could not confirm with the author, and any place where you deviated from this brief and why.
