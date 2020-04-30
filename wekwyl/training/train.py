import datetime
import functools
import json
import os

from wekwyl.models import unet
from wekwyl.models import utils
from wekwyl.training import datasets
from wekwyl.training import handlers
from wekwyl.training import losses
from wekwyl.training import transforms
from wekwyl.training import lr_schedulers

import numpy as np
from PIL import Image
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision

from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import Engine
from ignite.engine import Events
from ignite.handlers import EarlyStopping
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer


def add_tensorboard_handler(trainer, validator, model, optimizer, log_dir):
    # For details see: https://pytorch.org/ignite/contrib/handlers.html#tensorboard-logger

    # Create a logger
    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Attach the logger to the trainer to log training loss at each iteration
    tb_logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag='training',
            output_transform=lambda loss: loss,
        ),
        event_name=Events.ITERATION_COMPLETED,
    )

    # Attach the logger to the evaluator on the validation dataset and log NLL, Accuracy metrics after
    # each epoch. We setup `global_step_transform=global_step_from_engine(trainer)` to take the epoch of the
    # `trainer` instead of `evaluator`.
    tb_logger.attach(
        validator,
        log_handler=OutputHandler(
            tag='validation',
            output_transform=lambda loss: loss,
            global_step_transform=global_step_from_engine(trainer)
        ),
        event_name=Events.EPOCH_COMPLETED,
    )

    # Attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at each iteration
    tb_logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )

    # Attach the logger to the trainer to log model's weights norm after each iteration
    tb_logger.attach(
        trainer,
        log_handler=WeightsScalarHandler(model),
        event_name=Events.ITERATION_COMPLETED,
    )

    # Attach the logger to the trainer to log model's weights as a histogram after each epoch
    tb_logger.attach(
        trainer,
        log_handler=WeightsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED,
    )

    # Attach the logger to the trainer to log model's gradients norm after each iteration
    tb_logger.attach(
        trainer,
        log_handler=GradsScalarHandler(model),
        event_name=Events.ITERATION_COMPLETED,
    )

    # Attach the logger to the trainer to log model's gradients as a histogram after each epoch
    tb_logger.attach(
        trainer,
        log_handler=GradsHistHandler(model),
        event_name=Events.EPOCH_COMPLETED,
    )

    # We need to close the logger with we are done
    tb_logger.close()


def add_handlers(
    trainer, validator, train_loader, validation_loader, model, optimizer, config,
):
    training_saver = ModelCheckpoint(
        dirname=os.path.join(config.experiment_name, 'checkpoint'),
        filename_prefix='ckpt',
        n_saved=5,
        require_empty=False,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.ckpt_interval),
        training_saver,
        {'model': model, 'optimizer': optimizer},
    )

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    train_images_dir = os.path.join(
        config.experiment_name, 'images', 'train',
    )
    os.makedirs(train_images_dir)

    def get_frame(engine):
        return Image.fromarray(
            np.rint(
                engine.state.frame.transpose(1, 2, 0) * 255
            ).astype(np.uint8)
        )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.dump_interval),
        handlers.dump_image,
        folder=train_images_dir,
        get_image_fn=get_frame,
        suffix='frame',
    )

    def get_saliency(engine):
        return Image.fromarray(
            np.rint(engine.state.saliency[0] * 255).astype(np.uint8)
        )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.dump_interval),
        handlers.dump_image,
        folder=train_images_dir,
        get_image_fn=get_saliency,
        suffix='saliency',
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.vld_every_epoch),
        handlers.copy_dir,
        src=config.experiment_name,
        dst=os.path.join(config.experiments_dir, config.experiment_name),
    )

    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        handlers.reset_accumulators,
    )

    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        handlers.reset_accumulators,
    )

    trainer.add_event_handler(
        Events.EPOCH_STARTED,
        handlers.log_epoch,
    )

    trainer.add_event_handler(
        Events.ITERATION_STARTED(every=config.log_interval),
        handlers.log_iteration,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_interval),
        handlers.log_metrics,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        handlers.update_accumulators,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.log_interval),
        handlers.log_learning_rate,
        optimizer=optimizer,
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config.vld_every_epoch),
        handlers.run_on_validation,
        validation_engine=validator,
        validation_loader=validation_loader,
    )

    def get_nss(engine):
        return engine.state.mean_losses['NSS'].item()

    handler = EarlyStopping(
        patience=10,
        score_function=get_nss,
        trainer=trainer,
    )
    validator.add_event_handler(Events.COMPLETED, handler)

    best_model_saver = ModelCheckpoint(
        dirname=os.path.join(config.experiment_name, 'best_models'),  
        filename_prefix='best',
        score_name='nss',  
        score_function=get_nss,
        n_saved=config.best_model_count,
        require_empty=False
    )

    validator.add_event_handler(
        Events.COMPLETED,
        best_model_saver,
        {'model': model},
    )

    validation_images_dir = os.path.join(
        config.experiment_name, 'images', 'validation',
    )
    os.makedirs(validation_images_dir)

    validator.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.dump_interval),
        handlers.dump_image,
        folder=validation_images_dir,
        get_image_fn=get_frame,
        suffix='frame',
    )

    validator.add_event_handler(
        Events.ITERATION_COMPLETED(every=config.dump_interval),
        handlers.dump_image,
        folder=validation_images_dir,
        get_image_fn=get_saliency,
        suffix='saliency',
    )

    validator.add_event_handler(
        Events.EPOCH_STARTED,
        handlers.reset_accumulators,
    )

    validator.add_event_handler(
        Events.EPOCH_COMPLETED,
        handlers.log_metrics,
        reset_after_log=False,
    )

    validator.add_event_handler(
        Events.ITERATION_COMPLETED,
        handlers.update_accumulators,
    )


def make_loop_step_function(model, criterions, weights, lr_scheduler, device):
    def run_on_batch(engine, batch, train=True):
        if train:
            model.train()
            lr_scheduler.optimizer.zero_grad()
        else:
            model.eval()

        engine.state.batch_size = len(batch['frame'])

        frames = batch['frame'].to(device)
        maps = batch['saliency'].to(device)
        fixations = batch['fixations']

        maps_pred = model(frames)

        losses = {
            key: criterion(maps_pred, maps, fixations)
            for key, criterion in criterions.items()
        }
        engine.state.batch_losses = losses

        loss = sum(w * losses[key] for key, w in weights.items())
        if train:
            loss.backward()
            lr_scheduler.optimizer.step()
            lr_scheduler.step()

        engine.state.frame = frames[0].detach().cpu().numpy()
        engine.state.saliency = maps_pred[0].detach().cpu().numpy()

        output_losses = {
            key: value.item()
            for key, value in losses.items()
        }
        output_losses['loss'] = loss.item()
        return output_losses

    return run_on_batch


def make_dataloader(config, videos, transform, shuffle):
    dataset = datasets.SaliencyDataset(
        videos,
        config.dataset_dir,
        config.frames_folder,
        config.maps_folder,
        config.fixations_filename,
        transform,
        config.frac,
    )
    return data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=datasets.collate,
        pin_memory=config.pin_gpu_memory
    )


def run_train(config):
    if config.random_seed:
        print(f'Set random state: {config.random_seed}')
        np.random.seed(config.random_seed)
        th.manual_seed(config.random_seed)

    use_cuda = config.use_cuda and th.cuda.is_available()
    device = th.device('cuda' if use_cuda else 'cpu')
    print(f'Detected device: {device.type}')

    transform = torchvision.transforms.Compose(
        [
            transforms.ToChannelsFirst(),
            transforms.CastImages('float32'),
            transforms.NormalizeImages(),
            transforms.ToTensor(device),
        ]
    )

    print('Creating dataloaders...')

    trn_loader = make_dataloader(
        config, config.trn_videos, transform, shuffle=True,
    )
    print(f'Num training batches: {len(trn_loader)}')

    vld_loader = make_dataloader(
        config, config.vld_videos, transform, shuffle=False,
    )
    print(f'Num validation batches: {len(vld_loader)}')

    print('Creating model...')
    norm_layer = utils.get_norm_layer(config.norm_type)
    model = unet.CylindircUnetNN(
        input_nc=3,
        output_nc=1,
        num_downs=config.num_downsamples,
        ngf=config.num_filters,
        norm_layer=norm_layer,
        use_dropout=config.use_dropout,
    )
    model = model.to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.lr,
        nesterov=config.use_nesterov,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    print(f'Optimizer: {optimizer}')

    lr_scheduler = lr_schedulers.CosineAnnealingWithWarmupLR(
        optimizer,
        warmup_epochs=config.warmup_steps,
        T_max=len(trn_loader) * config.num_epochs
    )
    print('LR Scheduler: {}'.format(lr_scheduler.state_dict()))

    nss = losses.SphericalNSS(config.height, config.width).to(device)
    cc = losses.SphericalCC(config.height, config.width).to(device)
    mse = losses.SphericalMSE(config.height, config.width).to(device)

    criterions = {
        'NSS': lambda maps_pred, maps, fixations: nss(maps_pred, fixations),
        'CC': lambda maps_pred, maps, fixations: cc(maps_pred, maps),
        'MSE': lambda maps_pred, maps, fixations: mse(maps_pred, maps),
    }
    weights = {
        'NSS': config.nss_weight,
        'CC': config.cc_weight,
        'MSE': config.mse_weight,
    }

    run_on_batch = make_loop_step_function(model, criterions, weights, lr_scheduler, device)

    trainer = Engine(functools.partial(run_on_batch, train=True))
    validator = Engine(functools.partial(run_on_batch, train=False))

    print('Adding handlers to engines...')
    add_handlers(
        trainer, validator, trn_loader, vld_loader, model, lr_scheduler, config,
    )

    print('Adding TensorBoard handler...')
    add_tensorboard_handler(
        trainer,
        validator,
        model,
        optimizer,
        os.path.join(config.experiment_name, 'tb_log'),
    )

    print('Start training!')
    trainer.run(trn_loader, max_epochs=config.num_epochs)

    print('Done!')
