# Tomato leaf disease course experiment

Reproducible comparison of a small CNN, ImageNet-pretrained ResNet18, and the same ResNet18 with training-time augmentation. This is a course-scale comparison, not a novel architecture or a validated field diagnostic tool.

## Data

Original source: https://github.com/spMohanty/PlantVillage-Dataset at commit `7f7ecc7e1eaca78107e3affe7cb5abd9427e139a`.

Only seven tomato classes with image IDs matched to `leaf-map.json` are used. Target spot and tomato mosaic virus are excluded because their filenames cannot be matched by this lookup. Spider-mite injury is excluded as a pest category. Each retained class contributes 500 images (sampling seed 2026). Dataset leaf identities and exact-pixel duplicate groups are kept together in a 70/15/15 group-level split. Actual image counts are recorded in `data_summary.json`. Images are not redistributed here.

## Reproduction

Python 3.10, PyTorch 2.3.0, torchvision 0.18.0, NumPy 1.26 and Pillow 9.4. CUDA build 11.8. Use a separate environment instead of modifying shared packages.

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git source_metadata
git -C source_metadata fetch --depth 1 origin 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
git -C source_metadata update-ref HEAD 7f7ecc7e1eaca78107e3affe7cb5abd9427e139a
python prepare_experiment.py
CUDA_VISIBLE_DEVICES=0 python train_experiment.py
```

The metadata clone is intentionally not checked out; only selected files are downloaded. Training uses three seeds (2026/2027/2028), ten epochs and 128-pixel inputs. Checkpoints are selected only by validation macro-F1, then evaluated on the held-out test set. The two ResNet18 variants share architecture, initialization source, learning rate and training budget; only augmentation changes. The small CNN has different initialization and learning rate, so that contrast cannot isolate a single cause.

Blur and darkness tests are synthetic perturbations, not external field validation. Across-seed standard deviations measure optimization variation conditional on the fixed split, not uncertainty over new farms or collection sites.

## Outputs

`manifest.json` records original image paths, SHA256 checksums, leaf identities and frozen splits. Each `runs/<method>_seed<seed>/` contains configuration, epoch history, selected checkpoint, test metrics and predictions. Checkpoints are release assets rather than source commits.

## Attribution

Data and original labels belong to the PlantVillage authors. The author-maintained Hugging Face card labels the dataset CC BY-SA 3.0: https://huggingface.co/datasets/mohanty/PlantVillage . This project does not relicense the dataset or redistribute original images. ResNet18 uses torchvision's ImageNet-pretrained weights; applicable upstream model restrictions remain relevant. Original scripts are provided for educational reproduction. No crop treatment advice is given.

## Actual results

All nine formal runs completed. The fixed split contains 2,448 training, 512 validation and 540 test images, representing 1,799 disjoint connected leaf-identity groups in total. Clean accuracy (three-seed mean ± sample SD): Small CNN **92.35 ± 0.47%**; ResNet18 **96.91 ± 0.57%**; augmented ResNet18 **97.65 ± 1.30%**. Augmentation changes macro-F1 by **+0.70 ± 1.65 percentage points** on clean images and **+14.38 ± 2.84 points** on darkened images; blur gains are inconsistent across seeds. Read `results_summary.json` and `metrics_source.csv` for unrounded results and paired values. No significance claim is made.

`python audit_experiment.py` verifies pairwise split disjointness, atomic leaf IDs, all image checksums, and recomputes metrics from all 27 prediction CSV files. `python predict.py --checkpoint runs/resnet18_aug_seed2026/best.pt --image /path/to/image.jpg` runs CPU inference; load only trusted checkpoints. Data preparation and inference do not require changes to the user's SSH or account configuration.

AI assistance was used for code generation, experiment orchestration, analysis and manuscript drafting. The experiments were actually executed, and numerical results were recomputed from recorded predictions. This does not substitute for the student's review, understanding or any course-specific disclosure requirements.

## Paper and checkpoints

The release contains an editable bilingual two-column DOCX, its checked four-page PDF and `trained_checkpoints.tar.gz` (all nine validation-selected checkpoints). Extract the checkpoint archive at the repository root to restore the `runs/<method>_seed<seed>/best.pt` paths. The archive SHA256 is `92285ee5a12593adb4eb6b98fa38fec6dead5164411951bf3ada73891f3fbe46`.

`verify_checkpoint.py` reloads all nine checkpoints on the original CUDA backend and checks all clean-set predictions against the recorded CSVs. All nine pass. Floating-point implementations differ between CPU and CUDA; bitwise identity and identical decisions for borderline cases across backends are not guaranteed.

For paper regeneration, install `requirements-paper.txt`, run `python analyze_results.py` and `python build_paper.py`, then export the DOCX using a Word-compatible renderer. `render_word.ps1` uses Microsoft Word on Windows. The optional external figure layout auditor is loaded only when `FIGURE_QA_SCRIPTS` is configured; absence is explicitly reported rather than claimed as a successful audit.

S. P. Mohanty, D. P. Hughes, M. Salathé. Using Deep Learning for Image-Based Plant Disease Detection. Frontiers in Plant Science 7:1419, 2016. https://doi.org/10.3389/fpls.2016.01419
