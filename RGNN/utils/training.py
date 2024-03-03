#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import os
import schnetpack as spk
import torch
from torch.optim import Adam

from RGNN.train.trainer import Trainer
from RGNN.train.hooks.stop import NaNStoppingHook

__all__ = ["get_metrics", "get_trainer", "simple_loss_fn"]


def get_metrics(args):
    metrics = []
    if args.predict_property["forces"] is not None:
        metrics.append(
            spk.train.metrics.MeanAbsoluteError(
                args.predict_property["forces"],
                args.predict_property["forces"],
                element_wise=True,
            )
        )
        metrics.append(
            spk.train.metrics.RootMeanSquaredError(
                args.predict_property["forces"],
                args.predict_property["forces"],
                element_wise=True,
            ),
        )
    return metrics


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    #trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    #optimizer = Adam(trainable_params, lr=args.lr)
    optimizer = Adam(model.parameters(),lr=args.lr)
    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    # logger
    logger = spk.train.CSVHook(
        os.path.join(args.modelpath, "log"),
        metrics,
        every_n_epochs=args.log_every_n_epochs,
    )
    hooks.append(logger)

    # early stopping for nan
    hooks.append(NaNStoppingHook())

    # setup loss function
    loss_fn = simple_loss_fn(args)

    # setup trainer
    trainer = Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
        regularization=args.regularization,
    )
    return trainer


def simple_loss_fn(args):
    def loss(batch, result):
        if args.predict_property["forces"] is not None:
            property_name = args.predict_property["forces"]
        diff = batch[property_name] - result[property_name]
        diff = diff**2
        err_sq = torch.mean(diff)
        return err_sq

    return loss
