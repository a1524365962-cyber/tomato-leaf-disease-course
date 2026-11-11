# Figure evidence contract

Python/matplotlib is used throughout the existing Python training workflow. Target: single-column IEEE-like course-paper figures, width 3.35 inches; minimum text 5 pt. All plots use actual run artifacts, with PDF/SVG/400-dpi PNG exports and adjacent source CSV.

1. Validation curve: compare optimization trajectories. Three-seed mean and sample SD, fixed data split. It does not establish external generalization or training-time significance.
2. Confusion matrix: identify class-specific clean-test mistakes. One prespecified seed (2026), count scale, all seven classes shown. No uncertainty because it is one model's full test prediction matrix, not an aggregate.
3. Augmentation effect: compare paired macro-F1 differences for augmentation versus the same ResNet18 without augmentation under clean/blur/dark tests. All three paired seed values shown; square and error bars indicate mean and sample SD. No hypothesis test or population confidence interval is claimed with three seeds. The three conditions reuse the same held-out source images and are not independent cohorts.

All classes and all complete runs are retained. Data exclusions occur upstream for metadata integrity and task scope, not to improve figure aesthetics. Finalized artifacts must pass alignment/glyph/collision checks and visual inspection; automated layout checks alone do not verify the scientific claim.

## Method illustration v1.2

The three-panel overview uses a 175.3-mm cross-column layout. Panel a documents the two ResNet18 data paths, frozen split counts, augmentation-only branch, common normalization, logits, training objective and validation checkpoint selection. Panel b depicts correct C x H x W dimensions with schematic stacked planes, not activation measurements. Panel c separates the identity shortcut, addition and ReLU and discloses projection shortcuts at downsampling stage entries. Real leaf thumbnails come from the already selected PlantVillage training examples; derived augmentation views do not introduce new experimental observations. Export SVG/PDF with editable/vector text and PNG/TIFF at 600 dpi. No numerical result, model architecture or weight changes are implied.
