from collections import defaultdict
import os
import shutil as sh


def log_epoch(engine):
    print(f'Epoch {engine.state.epoch}/{engine.state.max_epochs}')


def log_iteration(engine):
    iteration = 1 + (engine.state.iteration - 1) % len(engine.state.dataloader)
    print(f'[Iter {iteration}/{len(engine.state.dataloader)}]')


def copy_dir(engine, src, dst):
    sh.copytree(src, dst)


def dump_image(engine, folder, get_image_fn, suffix='img'):
    it = engine.state.iteration
    path = os.path.join(folder, f'iter={it}_{suffix}.png')
    image = get_image_fn(engine)
    image.save(path)


def log_learning_rate(engine, optimizer):
    lrs = optimizer.get_lr()
    if len(lrs) == 1:
        print('Learning rate: {:.5f}'.format(lrs[0]))
    else:
        for i, lr in enumerate(lrs):
            print('Learning rate (group {}): {:.5f}'.format(i, lr))


def log_metrics(engine, reset_after_log=True):
    sorted_keys = sorted(engine.state.mean_losses.keys())
    titles = ['metrics'] + sorted_keys
        
    header = ' | '.join(['{:^14}'.format(title) for title in titles])
    print(header)

    losses_message = ' | '.join([
        '{:>14.4f}'.format(engine.state.mean_losses[key])
        for key in sorted_keys
    ])
    print('{:<14} | {}'.format('Loss', losses_message))

    if reset_after_log:
        reset_accumulators(engine)


def reset_accumulators(engine):
    engine.state.total_items = 0
    engine.state.mean_losses = defaultdict(float)


def update_accumulators(engine):
    '''Average value computation using Welford method.
    See: https://habr.com/ru/post/333426
    '''
    engine.state.total_items += engine.state.batch_size
    proportion = engine.state.batch_size / engine.state.total_items

    for key in engine.state.output.keys():
        engine.state.mean_losses[key] += (
            proportion *
            (engine.state.output[key] - engine.state.mean_losses[key])
        )


def run_on_validation(engine, validation_engine, validation_loader):
    print('[Validation]')
    validation_engine.run(validation_loader)
