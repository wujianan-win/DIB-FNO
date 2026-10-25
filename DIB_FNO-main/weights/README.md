# Checkpoints

Binary checkpoint files are stored with Git LFS. Run `git lfs pull` after cloning.

## Reported models

`checkpoints/` contains three seeds for each trained configuration:

- `dibfno_h16_seed{0,1,2}.pt`: DIB-FNO, hidden width 16, one block, 3,106 parameters.
- `static_fno_h17_seed{0,1,2}.pt`: learned static gate, hidden width 17, one block, 3,173 parameters.
- `fixed_fno_h16_seed{0,1,2}.pt`: same-width fixed FNO, hidden width 16, one block, 1,896 parameters.

`frozen_mean_gate/` contains portable DIB-FNO backbones paired with train-only mean gates for the post hoc frozen-gate ablation.

All reported training runs used 15 epochs. The checkpoint payload includes model type, model configuration, selected epoch, seed, channel order, normalization, and state dictionary. Load checkpoints through the evaluation CLI rather than instantiating an assumed architecture manually.

These models are research artifacts for the documented 5.625° eight-channel data contract. They are not operational weather products.
