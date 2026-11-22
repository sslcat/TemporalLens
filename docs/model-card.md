# Model card

I trained TemporalLens v1 to label still, left, right, up and down for one cyan marker. The released predictor is a 1,637-parameter five-frame temporal CNN with a fixed two-coordinate pixel encoder. The companion 1,317-parameter frame checkpoint provides the experimental comparison. Both are PyTorch state dictionaries. The release loads the architecture from source and reads tensor weights with `weights_only=True`.

I trained from random initialization with weighted frame-level cross-entropy on 240 generated clips and selected checkpoints on 60 validation clips. The lowest validation loss occurred at epoch 13 for the frame head and 140 for the temporal head. I evaluated the selected heads once in the preregistered comparison on 96 additional scenes. [Training](training.md) · [Dataset](dataset-card.md) · [Results and interventions](evaluation.md)

The supported input is a continuous view of one visible, bright cyan marker undergoing cardinal translations with pauses against a dark background. The model can identify runs outside the two-event training pattern, but those patterns have no benchmark coverage. A geometric marker-quality check rejects some unsupported inputs. It cannot establish semantic compatibility for every uploaded scene. A similarly colored patch, an edit, a diagonal trajectory or a camera pan can yield confident incorrect events. The softmax display score is uncalibrated. I have no evidence for human action recognition, open-world event detection, planning or causal reasoning.

I chose MIT for the original checkpoints and generated assets. No pretrained weights or external video dataset enter the model. I used AI coding assistance for implementation, experiment execution and documentation. [Attribution](../NOTICE.md)
