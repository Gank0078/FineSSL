"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import os
import pickle
import shutil
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "resume_from_checkpoint",
]


def save_checkpoint(
    state,
    save_dir,
    is_best=False,
    remove_module_from_keys=True,
    model_name=""
):
    r"""Save checkpoint.

    Args:
        state (dict): dictionary.
        save_dir (str): directory to save checkpoint.
        is_best (bool, optional): if True, this checkpoint will be copied and named
            ``model-best.pth.tar``. Default is False.
        remove_module_from_keys (bool, optional): whether to remove "module."
            from layer names. Default is True.
        model_name (str, optional): model name to save.
    """
    os.makedirs(save_dir, exist_ok=True)

    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict

    # save model
    epoch = state["epoch"]
    if not model_name:
        model_name = "model.pth.tar-" + str(epoch)
    fpath = osp.join(save_dir, model_name)
    torch.save(state, fpath)
    print(f"Checkpoint saved to {fpath}")

    # save current model name
    checkpoint_file = osp.join(save_dir, "checkpoint")
    checkpoint = open(checkpoint_file, "w+")
    checkpoint.write("{}\n".format(osp.basename(fpath)))
    checkpoint.close()

    if is_best:
        best_fpath = osp.join(osp.dirname(fpath), "model-best.pth.tar")
        shutil.copy(fpath, best_fpath)
        print('Best checkpoint saved to "{}"'.format(best_fpath))


def load_checkpoint(fpath, device=None):
    r"""Load checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError("File path is None")

    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))

    # map_location = None if torch.cuda.is_available() else "cpu"
    # device = torch.device("cuda:{}".format(0))
    # device = torch.device("cuda")
    map_location = device

    try:
        checkpoint = torch.load(fpath, map_location=map_location)

    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )

    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise

    return checkpoint


def resume_from_checkpoint(fdir, model, optimizer=None, scheduler=None):
    r"""Resume training from a checkpoint.

    This will load (1) model weights and (2) ``state_dict``
    of optimizer if ``optimizer`` is not None.

    Args:
        fdir (str): directory where the model was saved.
        model (nn.Module): model.
        optimizer (Optimizer, optional): an Optimizer.
        scheduler (Scheduler, optional): an Scheduler.

    Returns:
        int: start_epoch.

    Examples::
        >>> fdir = 'log/my_model'
        >>> start_epoch = resume_from_checkpoint(fdir, model, optimizer, scheduler)
    """
    with open(osp.join(fdir, "checkpoint"), "r") as checkpoint:
        model_name = checkpoint.readlines()[0].strip("\n")
        fpath = osp.join(fdir, model_name)

    print('Loading checkpoint from "{}"'.format(fpath))
    checkpoint = load_checkpoint(fpath)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    # model.load_state_dict(checkpoint["state_dict"])
    if 'squeue' in checkpoint:
        squeue = checkpoint['squeue']
    else:
        squeue = None
    print("Loaded model weights")

    # if optimizer is not None and "optimizer" in checkpoint.keys():
    #     optimizer.load_state_dict(checkpoint["optimizer"])
    #     print("Loaded optimizer")
    #
    # if scheduler is not None and "scheduler" in checkpoint.keys():
    #     scheduler.load_state_dict(checkpoint["scheduler"])
    #     print("Loaded scheduler")

    start_epoch = checkpoint["epoch"]
    print("Previous epoch: {}".format(start_epoch))

    return start_epoch
