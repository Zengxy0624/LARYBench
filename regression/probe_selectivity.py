"""
Stage-0 选择性探针 (selectivity probe) —— 支持无人值守过夜批跑
================================================================
问题:冻结的 dinov2 patch 特征里,动作相关信息是否集中在一个*低维、特定*的子空间?

做法:在每个 token 上加一个 1024 -> k 的投影(在 256 个 token 间共享,**保留空间网格**,
不做 mean-pool),再喂进 LARY 原版 MLPResNet 回归头。比较两种同容量(同 k)条件:
  - lowrank : 投影可学习,被 action loss 驱动 -> 找 action-aligned 方向
  - random  : 投影随机冻结 -> 对照,衡量"随便 k 个方向"能拿到多少

读数:lowrank 比 random 好多少、随 k 怎么收敛。
  - 小 k 下 lowrank >> random 且 random 追不上 -> 信息集中在特定方向 (更 selective)
  - random 很快追平 -> 信息分散,谈不上"选择"

不跑任何 VFM,只读 extract 阶段缓存的 .npz。训练极快 (<1s/epoch),瓶颈是一次性读盘建缓存。

过夜批跑设计(安全栏):
  * 缓存只建一次(大池子),每个 seed 在内存重抽子集 -> 零额外读盘
  * 每个条件跑完**立即**把结果 append 进 JSONL -> 半夜崩了也有部分结果
  * seed 外层循环 -> 跑完一个 seed 就有完整曲线,后续 seed 只收窄误差棒
  * 早停 + 每条件墙钟上限 + 全局时间预算 -> 绝不跑过夜、不被卡死
  * 可断点续跑:重启自动跳过 JSONL 里已完成的 (seed,mode,k)

用法(在 baselines/lary 下,先 source ../../scripts/lary_env.sh):
  python -m regression.probe_selectivity \
    --n-pool-train 24000 --n-pool-val 4000 --n-train-sub 8000 \
    --seeds 0 1 2 3 4 --ks 2 4 8 16 32 64 128 --modes lowrank random \
    --max-epochs 100 --patience 15 --max-hours 7
"""
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from regression.main import (
    ActionExpertDataset, MLPResNet,
    get_action_dim, get_action_steps, get_dim_labels, get_group_indices,
)

CALVIN_MEAN = np.array([0.03993005, -0.1113833, 0.50033228, 1.04580053, -0.08165425, 1.58390577, -0.08441296])
CALVIN_STD = np.array([0.14403107, 0.09919957, 0.05518382, 2.89455128, 0.13053949, 0.57474015, 0.99643086])

# 切换 VFM 就改这一处:dinov2-origin / dinov3-origin / ...(对应 extract 产出的 CSV 名)
LA_MODEL = "dinov2-origin"


class SelectivityProbe(nn.Module):
    """per-token 投影 (in_dim->k, 跨 token 共享) -> flatten -> MLPResNet。
    mode='full' 时跳过投影(k=in_dim),即未压缩基准(显存大,慎用)。"""
    def __init__(self, n_frames, n_tokens, in_dim, k, out_dim, mode, hidden=4096, blocks=2):
        super().__init__()
        self.n_frames, self.n_tokens, self.in_dim = n_frames, n_tokens, in_dim
        self.mode = mode
        if mode == 'full':
            self.proj = None
            flat_dim = n_frames * n_tokens * in_dim
        else:
            self.proj = nn.Linear(in_dim, k, bias=False)
            if mode == 'random':
                for p in self.proj.parameters():
                    p.requires_grad_(False)
            flat_dim = n_frames * n_tokens * k
        self.mlp = MLPResNet(blocks, flat_dim, hidden, out_dim)

    def forward(self, x):  # x: (B, n_frames*n_tokens*in_dim)
        B = x.shape[0]
        if self.proj is not None:
            x = x.view(B, self.n_frames, self.n_tokens, self.in_dim)
            x = self.proj(x)
            x = x.reshape(B, -1)
        return self.mlp(x)


def build_cache(dataset, n, seed, tag):
    """随机抽 n 个样本,把 (feature_fp16, action_fp32) 读进内存(一次性读盘)。
    预分配再逐行写入,避免 list+stack 的峰值翻倍。"""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    f0, a0 = dataset[int(idx[0])][0], dataset[int(idx[0])][1]
    X = torch.empty((len(idx), f0.numel()), dtype=torch.float16)
    Y = torch.empty((len(idx), a0.numel()), dtype=torch.float32)
    X[0], Y[0] = f0.to(torch.float16), a0
    t0 = time.time()
    for j in range(1, len(idx)):
        f, a = dataset[int(idx[j])][0], dataset[int(idx[j])][1]
        X[j], Y[j] = f.to(torch.float16), a
        if (j + 1) % 2000 == 0:
            print(f"  [{tag}] cached {j+1}/{len(idx)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"  [{tag}] done {len(idx)} in {time.time()-t0:.0f}s  "
          f"({X.element_size()*X.nelement()/1e9:.1f} GB fp16)", flush=True)
    return X, Y


def load_or_build_pool(cache_dir, dataset, n, seed, tag):
    """有 .npy 缓存就 mmap 只读(多进程共享 page cache、内存只一份);否则建池,
    若给了 cache_dir 就落盘供后续进程 mmap。"""
    if cache_dir:
        xp = os.path.join(cache_dir, f'X_{tag}.npy')
        yp = os.path.join(cache_dir, f'Y_{tag}.npy')
        if os.path.exists(xp) and os.path.exists(yp):
            X = torch.from_numpy(np.load(xp, mmap_mode='r'))
            Y = torch.from_numpy(np.load(yp, mmap_mode='r'))
            print(f"  [{tag}] mmap 共享缓存 {tuple(X.shape)} <- {xp}", flush=True)
            return X, Y
    X, Y = build_cache(dataset, n, seed, tag)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, f'X_{tag}.npy'), X.numpy())
        np.save(os.path.join(cache_dir, f'Y_{tag}.npy'), Y.numpy())
        print(f"  [{tag}] 已落盘缓存 -> {cache_dir}", flush=True)
    return X, Y


def per_dim_and_group_mse(pred, target, action_steps, dim_labels, group_idx):
    num = len(dim_labels)
    p = pred.view(-1, action_steps, num)
    t = target.view(-1, action_steps, num)
    sq = (p - t) ** 2
    per_dim = sq.mean(dim=(0, 1))
    out = {'overall': sq.mean().item()}
    for i, lab in enumerate(dim_labels):
        out[lab] = per_dim[i].item()
    if group_idx:
        for g, ids in group_idx.items():
            out[g] = per_dim[ids].mean().item()
    return out


def train_eval(cond_mode, k, Xtr, Ytr, Xva, Yva_d, shape, action_steps, dim_labels,
               group_idx, out_dim, args, device):
    n_frames, n_tokens, in_dim = shape
    torch.manual_seed(hash((cond_mode, k, args._seed)) % (2**31))
    model = SelectivityProbe(n_frames, n_tokens, in_dim, k, out_dim, cond_mode,
                             hidden=args.hidden, blocks=args.blocks).to(device)
    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    n_tr = Xtr.shape[0]
    best, best_ep, bad = None, -1, 0
    t_cond = time.time()
    for ep in range(args.max_epochs):
        model.train()
        perm = torch.randperm(n_tr)
        for s in range(0, n_tr, args.batch_size):
            bi = perm[s:s + args.batch_size]
            xb = Xtr[bi].to(device).float()
            yb = Ytr[bi].to(device).float()
            pred = model(xb)
            loss = F.huber_loss(pred, yb, delta=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = [model(Xva[s:s + 1024].to(device).float()) for s in range(0, Xva.shape[0], 1024)]
            m = per_dim_and_group_mse(torch.cat(preds, 0), Yva_d, action_steps, dim_labels, group_idx)
        if best is None or m['overall'] < best['overall'] - 1e-5:
            best, best_ep, bad = m, ep, 0
        else:
            bad += 1
        elapsed = time.time() - t_cond
        if args.patience and bad >= args.patience:
            break
        if args.cond_time_cap and elapsed > args.cond_time_cap:
            print(f"  [{cond_mode} k={k}] hit cond-time-cap {args.cond_time_cap}s at ep{ep+1}", flush=True)
            break
    best['_params'] = n_param
    best['_best_epoch'] = best_ep + 1
    best['_ran_epochs'] = ep + 1
    best['_cond_sec'] = round(time.time() - t_cond, 1)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def load_done(jsonl):
    done = set()
    if os.path.exists(jsonl):
        with open(jsonl) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['seed'], r['mode'], r['k']))
                except Exception:
                    pass
    return done


def summarize(jsonl, group_idx, out_path):
    rows = []
    with open(jsonl) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    cols = ['overall'] + (list(group_idx.keys()) if group_idx else [])
    agg = {}
    for r in rows:
        key = (r['mode'], r['k'])
        agg.setdefault(key, {c: [] for c in cols})
        for c in cols:
            if c in r:
                agg[key][c].append(r[c])
    print("\n" + "=" * 78)
    print(f"{'condition':<16}{'n':>4}" + "".join(f"{c:>14}" for c in cols))
    print("-" * 78)
    summary = {}
    for (mode, k) in sorted(agg.keys(), key=lambda x: (x[0], x[1])):
        d = agg[(mode, k)]
        n = len(d['overall'])
        cell = []
        rec = {'mode': mode, 'k': k, 'n_seeds': n}
        for c in cols:
            mu = float(np.mean(d[c])) if d[c] else float('nan')
            sd = float(np.std(d[c])) if d[c] else float('nan')
            rec[c] = {'mean': mu, 'std': sd}
            cell.append(f"{mu:.3f}±{sd:.3f}")
        summary[f"{mode}_k{k}"] = rec
        print(f"{mode+'_k'+str(k):<16}{n:>4}" + "".join(f"{x:>14}" for x in cell))
    print("=" * 78)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[summary saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='calvin')
    ap.add_argument('--model', default=LA_MODEL, help='extract 产出的 VFM 名:dinov2-origin / dinov3-origin / ...')
    ap.add_argument('--stride', type=int, default=5)
    ap.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument('--modes', nargs='+', default=['lowrank', 'random'])
    ap.add_argument('--include-full', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--n-pool-train', type=int, default=24000, help='一次性读进内存的训练池')
    ap.add_argument('--n-pool-val', type=int, default=4000)
    ap.add_argument('--n-train-sub', type=int, default=8000, help='每 seed 从池里重抽的训练量;0=用整池')
    ap.add_argument('--max-epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=15, help='早停;0=关')
    ap.add_argument('--cond-time-cap', type=float, default=240, help='单条件墙钟上限秒;0=关')
    ap.add_argument('--max-hours', type=float, default=7.0, help='全局时间预算(小时),超了不再起新条件')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--hidden', type=int, default=4096)
    ap.add_argument('--blocks', type=int, default=2)
    ap.add_argument('--results-jsonl', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--pool-cache', default=None, help='池 .npy 缓存目录;存在则 mmap 共享读,否则建并落盘')
    ap.add_argument('--build-pool-only', action='store_true', help='只建池缓存后退出(供多卡进程随后 mmap)')
    args = ap.parse_args()

    assert args.dataset == 'calvin', "本探针先只接 calvin absolute"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = os.environ['PROJECT_ROOT']
    data_dir = Path(root) / 'baselines' / 'lary' / 'data'
    train_csv = str(data_dir / f"train_la_{args.dataset}_{args.stride}_{args.model}.csv")
    val_csv = str(data_dir / f"val_la_{args.dataset}_{args.stride}_{args.model}.csv")
    log_dir = Path(os.environ.get('LARY_LOG_DIR', '.')) / 'regression'
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.results_jsonl or str(log_dir / 'selectivity_results.jsonl')
    out = args.out or str(log_dir / 'selectivity_summary.json')

    action_dim = get_action_dim(args.dataset)
    action_steps = get_action_steps(args.stride, 'absolute')
    dim_labels = get_dim_labels(args.dataset)
    group_idx = get_group_indices(args.dataset)
    out_dim = action_dim * action_steps

    print(f"[setup] device={device} out_dim={out_dim} jsonl={jsonl}", flush=True)
    tr_ds = ActionExpertDataset(train_csv, args.dataset, args.stride, None,
                                CALVIN_MEAN, CALVIN_STD, action_mode='absolute')
    va_ds = ActionExpertDataset(val_csv, args.dataset, args.stride, None,
                                CALVIN_MEAN, CALVIN_STD, action_mode='absolute')
    raw = np.load(tr_ds.data.iloc[0]['la_path'])['tokens']
    shape = tuple(raw.shape)
    print(f"[setup] feature tokens shape = {shape}", flush=True)

    # 一次性建池(读盘) —— 训练池 + 固定验证池;给了 --pool-cache 则 mmap 共享
    print(f"[cache] pool train n={args.n_pool_train} ...", flush=True)
    Xpool, Ypool = load_or_build_pool(args.pool_cache, tr_ds, args.n_pool_train, 12345, 'train-pool')
    print(f"[cache] pool val n={args.n_pool_val} (固定验证集) ...", flush=True)
    Xva, Yva = load_or_build_pool(args.pool_cache, va_ds, args.n_pool_val, 54321, 'val-pool')
    if args.build_pool_only:
        print("[build-pool-only] 池缓存就绪,退出", flush=True)
        return
    # 验证特征留 CPU(fp16),eval 时逐 batch 上 GPU,省 ~8GB 显存;只有小的 Yva 常驻 GPU
    Yva_d = Yva.to(device).float()

    conds = [(m, k) for m in args.modes for k in args.ks]
    if args.include_full:
        conds.append(('full', int(shape[2])))
    done = load_done(jsonl)
    if done:
        print(f"[resume] 已完成 {len(done)} 个条件,跳过", flush=True)

    t_start = time.time()
    budget_s = args.max_hours * 3600
    fout = open(jsonl, 'a')
    stopped = False
    for seed in args.seeds:
        if stopped:
            break
        args._seed = seed
        # 每 seed 从池里重抽训练子集(内存内,零读盘)
        if args.n_train_sub and args.n_train_sub < Xpool.shape[0]:
            sub = np.random.default_rng(seed).choice(Xpool.shape[0], args.n_train_sub, replace=False)
            sub = torch.from_numpy(sub)
            Xtr, Ytr = Xpool[sub], Ypool[sub]
        else:
            Xtr, Ytr = Xpool, Ypool
        print(f"\n########## seed={seed}  train={Xtr.shape[0]} val={Xva.shape[0]} "
              f"(elapsed {(time.time()-t_start)/60:.0f}min) ##########", flush=True)
        for mode, k in conds:
            if (seed, mode, k) in done:
                continue
            if time.time() - t_start > budget_s:
                print(f"[budget] 到 {args.max_hours}h 上限,停止起新条件", flush=True)
                stopped = True
                break
            r = train_eval(mode, k, Xtr, Ytr, Xva, Yva_d, shape, action_steps,
                           dim_labels, group_idx, out_dim, args, device)
            rec = {'seed': seed, 'mode': mode, 'k': k, **r}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"  [{mode} k={k}] overall={r['overall']:.4f} "
                  f"pos={r.get('position',float('nan')):.4f} ori={r.get('orientation',float('nan')):.4f} "
                  f"grip={r.get('gripper',float('nan')):.4f}  "
                  f"best@ep{r['_best_epoch']}/{r['_ran_epochs']}  {r['_cond_sec']}s", flush=True)
    fout.close()
    print(f"\n[done] 总耗时 {(time.time()-t_start)/60:.0f}min", flush=True)
    summarize(jsonl, group_idx, out)


if __name__ == '__main__':
    main()
