"""Generate Burgers' train + test datasets only (skips the large sampling sets).

Writes:
    datasets/data/burgers_train_nIC80_nBC80.h5   (~260 MB)
    datasets/data/burgers_test_nIC30_nBC30.h5    (~36 MB)
"""
from datasets.generate_burgers1d_data import generate_burgers_dataset

generate_burgers_dataset(path="datasets/data/", N_ic=80, N_bc=80, seed=42,
                         filename="burgers_train")
generate_burgers_dataset(path="datasets/data/", N_ic=30, N_bc=30, seed=0,
                         filename="burgers_test")
