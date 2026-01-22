# Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution
# Requirements
We used `uv`. To set up the environment and install requirements:
```
uv sync
```
# To get Dataset:
We have included scripts we used to download data from the Johns Hopkins Turbulence Database. See the `dataset_creation/` folder. It took a few hours to get this data for us.

# Model weights:
See the `outputs/` folder for all saved models. The main file is `*checkpoint.pt`, which is saved during training:

```
# in `train.py`
torch.save(
    {
        "model": best_state_cpu,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "seed": config.seed
    },
    os.path.join(config['directory'], f'{config.barcode}_checkpoint.pt'),
)
```

There are also loss plots in these folders. `outputs/*/experiment_setup.txt` contains the config file used to run each experiment.

# Running models
You will need to first download the dataset using our scripts in `dataset_creation`. The main scripts are `src/train.py`, and `src/test_model.py`.

Examples:
```
uv run src/train.py config_name --device cuda:0
```

```
uv run src/test_model.py model_subdirectory --device cuda:0
```

I've included the scripts we used to run the experiments on slurm. If you have any questions, feel free to contact rmcconke@mit.edu. 


