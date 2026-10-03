"""Object-count-grouped GNN training -- the SAME training process as
train_gnn_dyn.py (identical loop body: multi-step rollout loss, optimizer,
logging, checkpointing), with only the data-loading plumbing swapped for
GroupedParticleDataset (see its module docstring for exactly what and why).

Kept as a separate file rather than editing train_gnn_dyn.py in place, per
the project decision to treat that script as the unmodified training-process
baseline (so results are never confounded by a changed training procedure)
and introduce new data support alongside it instead.

Two code-level differences from train_gnn_dyn.py, both inert w.r.t. training
results:
  - No throwaway FlexEnv reset to fetch camera params: GroupedParticleDataset
    needs no camera frame (see its docstring), so the `cam` argument is
    dropped from its constructor call entirely.
  - config path and TRAIN_ROOT point at this study's own config/output,
    config/train/gnn_dyn_grouped.yaml and data/gnn_dyn_model_grouped.

Usage (as run in the source repo):
    python -m train.train_gnn_dyn_grouped

Ported to pile_manipulation EXP-0064 (2026-10-03): only import paths and the
config/output paths changed (now CLI args; defaults reproduce the as-run
config). Run from the repo root:
    PYTHONPATH=. python -u experiments/EXP-0064-obj-count-effect-study/code/train_gnn_dyn_grouped.py \
        --config <resolved config.yaml> --out-root <dir>
The training loop body is untouched.
"""
import argparse
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # EXP-0064 code/ (ported adapter)
from dataset_grouped_particles import GroupedParticleDataset
from Baselines.GNN.model.gnn_dyn import PropNetDiffDenModel  # byte-identical to the source repo's model/gnn_dyn.py
from utils import set_seed, count_trainable_parameters, get_lr, AverageMeter, load_yaml, save_yaml, get_current_YYYY_MM_DD_hh_mm_ss_ms


def collate_fn(data):
    states, states_delta, attrs, particle_num, particle_den, color_imgs = zip(*data)
    max_len = max(particle_num)
    batch_size = len(data)
    n_time, _, n_dim = states[0].shape
    states_tensor = torch.zeros((batch_size, n_time, max_len, n_dim), dtype=torch.float32)
    states_delta_tensor = torch.zeros((batch_size, n_time - 1, max_len, n_dim), dtype=torch.float32)
    attr = torch.zeros((batch_size, n_time, max_len), dtype=torch.float32)
    particle_num_tensor = torch.tensor(particle_num, dtype=torch.int32)
    particle_den_tensor = torch.tensor(particle_den, dtype=torch.float32)
    color_imgs_np = np.array(color_imgs)
    color_imgs_tensor = torch.tensor(color_imgs_np, dtype=torch.float32)

    for i in range(len(data)):
        states_tensor[i, :, :particle_num[i], :] = states[i]
        states_delta_tensor[i, :, :particle_num[i], :] = states_delta[i]
        attr[i, :, :particle_num[i]] = attrs[i]

    return states_tensor, states_delta_tensor, attr, particle_num_tensor, particle_den_tensor, color_imgs_tensor


def train(config_path='experiments/EXP-0064-obj-count-effect-study/code/configs/gnn_dyn_grouped.yaml',
          train_root='data/gnn_dyn_model_grouped'):

    config = load_yaml(config_path)
    n_rollout = config['train']['n_rollout']
    n_history = config['train']['n_history']
    ckp_per_iter = config['train']['ckp_per_iter']
    log_per_iter = config['train']['log_per_iter']
    n_epoch = config['train']['n_epoch']

    set_seed(config['train']['random_seed'])

    use_gpu = torch.cuda.is_available()

    ### make log dir
    TRAIN_ROOT = train_root
    if config['train']['particle']['resume']['active']:
        TRAIN_DIR = os.path.join(TRAIN_ROOT, config['train']['particle']['resume']['folder'])
    else:
        TRAIN_DIR = os.path.join(TRAIN_ROOT, get_current_YYYY_MM_DD_hh_mm_ss_ms())
    os.system('mkdir -p ' + TRAIN_DIR)
    save_yaml(config, os.path.join(TRAIN_DIR, "config.yaml"))

    if not config['train']['particle']['resume']['active']:
        log_fout = open(os.path.join(TRAIN_DIR, 'log.txt'), 'w')
    else:
        log_fout = open(os.path.join(TRAIN_DIR, 'log_resume_epoch_%d_iter_%d.txt' % (
            config['train']['particle']['resume']['epoch'], config['train']['particle']['resume']['iter'])), 'w')

    ### dataloaders
    phases = ['train', 'valid']
    datasets = {phase: GroupedParticleDataset(config['train']['data_root'], config, phase) for phase in phases}

    dataloaders = {phase: DataLoader(
        datasets[phase],
        batch_size=config['train']['batch_size'],
        shuffle=True if phase == 'train' else False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn)
        for phase in phases}

    print('train/valid windows: %d / %d' % (len(datasets['train']), len(datasets['valid'])))

    ### create model
    model = PropNetDiffDenModel(config, use_gpu)
    print("model #params: %d" % count_trainable_parameters(model))

    # resume training of a saved model (if given)
    if config['train']['particle']['resume']['active']:
        model_path = os.path.join(TRAIN_DIR, 'net_epoch_%d_iter_%d.pth' % (
            config['train']['particle']['resume']['epoch'], config['train']['particle']['resume']['iter']))
        print("Loading saved ckp from %s" % model_path)

        pretrained_dict = torch.load(model_path)
        model.load_state_dict(pretrained_dict)

    if use_gpu:
        model = model.cuda()

    ### optimizer and losses
    params = model.parameters()
    optimizer = torch.optim.Adam(params,
                                lr=float(config['train']['lr']),
                                betas=(config['train']['adam_beta1'], 0.999))

    # start training
    st_epoch = config['train']['particle']['resume']['epoch'] if config['train']['particle']['resume']['epoch'] > 0 else 0
    best_valid_loss = np.inf

    for epoch in range(st_epoch, n_epoch):

        for phase in phases:
            model.train(phase == 'train')
            meter_loss = AverageMeter()

            for i, data in enumerate(dataloaders[phase]):

                states, states_delta, attrs, particle_nums, particle_dens, _ = data

                B, length, n_obj, _ = states.size()
                assert length == n_rollout + n_history

                if use_gpu:
                    states = states.cuda()
                    attrs = attrs.cuda()
                    states_delta = states_delta.cuda()
                    particle_dens = particle_dens.cuda()

                loss = 0.

                with torch.set_grad_enabled(phase == 'train'):

                    s_cur = states[:, 0]
                    a_cur = attrs[:, 0]

                    for idx_step in range(n_rollout):
                        s_nxt = states[:, idx_step + 1]
                        s_delta = states_delta[:, idx_step]

                        s_pred = model.predict_one_step(a_cur, s_cur, s_delta, particle_dens)

                        for j in range(B):
                            loss += F.mse_loss(s_pred[j, :particle_nums[j]], s_nxt[j, :particle_nums[j]])

                        s_cur = s_pred

                    loss = loss / (n_rollout * B)

                meter_loss.update(loss.item(), B)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                ### log and save ckp
                if i % log_per_iter == 0:
                    log = '%s [%d/%d][%d/%d] LR: %.6f, Loss: %.6f (%.6f)' % (
                        phase, epoch, n_epoch, i, len(dataloaders[phase]),
                        get_lr(optimizer),
                        np.sqrt(loss.item()), np.sqrt(meter_loss.avg))

                    print()
                    print(log)
                    log_fout.write(log + '\n')
                    log_fout.flush()

                if phase == 'train' and i % ckp_per_iter == 0:
                    torch.save(model.state_dict(), '%s/net_epoch_%d_iter_%d.pth' % (TRAIN_DIR, epoch, i))

            log = '%s [%d/%d] Loss: %.6f, Best valid: %.6f' % (
                phase, epoch, n_epoch, np.sqrt(meter_loss.avg), np.sqrt(best_valid_loss))
            print(log)
            log_fout.write(log + '\n')
            log_fout.flush()

            if phase == 'valid':
                if meter_loss.avg < best_valid_loss:
                    best_valid_loss = meter_loss.avg
                    torch.save(model.state_dict(), '%s/net_best.pth' % (TRAIN_DIR))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='experiments/EXP-0064-obj-count-effect-study/code/configs/gnn_dyn_grouped.yaml')
    ap.add_argument('--out-root', default='data/gnn_dyn_model_grouped')
    args = ap.parse_args()
    train(args.config, args.out_root)
