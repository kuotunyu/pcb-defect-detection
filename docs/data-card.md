# HRIPCB data card

## Dataset used

- Upstream identifier: `akhatova/pcb-defects` (Kaggle distribution of HRIPCB / PKU-Market-PCB)
- Observed structure: 693 JPEG images, 693 Pascal VOC annotations, 2,953 boxes
- Classes: `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`,
  `spurious_copper`
- Group unit: leading filename board id; ten non-contiguous board ids

The immutable identity used by this experiment is the normalized image/label content fingerprint
`8e5f0c880af67019bfc7ab5b08a4e63cc33726c97b5a77a41ebb27ddb3709ed4`. Counts alone are not
accepted as dataset identity.

## Intended use

Controlled research and portfolio demonstration of PCB object detection, split leakage, evaluation
protocols, and deployment gates. This dataset is not evidence of factory-line performance.

## Split policy

Board 08 supplies disjoint final-test and leaky-exposure siblings. Board 01 supplies disjoint
validation and calibration partitions. All other boards form the grouped training baseline. The
leaky control replaces exactly five images per class, preserving the baseline train size and class
histogram. See the frozen manifest for exact stems and hashes.

## Known limitations

- Only ten template boards; the common final test uses one board.
- Defects are not representative of every fabrication process, imaging stack, or factory domain.
- Image resampling cannot estimate variance across unseen boards.
- Filename board ids are treated as the grouping unit; no stronger physical-board provenance is
  available in the observed distribution.

## License status

The project has not verified an explicit upstream dataset license. Dataset pixels and pixel-derived
examples are therefore excluded from the candidate source tree. Users must independently establish
rights before downloading, training, redistributing images, or releasing weights.
