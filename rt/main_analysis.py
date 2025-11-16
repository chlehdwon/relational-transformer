import os
import random
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.distributed as dist
import wandb
from sklearn.metrics import r2_score, roc_auc_score
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grads_with_norm_, get_total_norm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rt.data import RelationalDataset
from rt.model import RelationalTransformer


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def all_gather_nd(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Gathers tensor arrays of different lengths in a list.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    All the other dimensions should be equal between the tensors.
    Adapted from: https://stackoverflow.com/a/71433508

    Args:
        tensor (Tensor): Tensor to be broadcast from current process.

    Returns:
        list[Tensor]: List of tensors gathered from all processes.
    """
    world_size = dist.get_world_size()
    local_size = torch.tensor(tensor.size(), device=tensor.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_length = max(size[0] for size in all_sizes)

    length_diff = max_length.item() - local_size[0].item()
    if length_diff:
        pad_size = (length_diff, *tensor.size()[1:])
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding))

    all_tensors_padded = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(all_tensors_padded, tensor)
    all_tensors = []
    for tensor_, size in zip(all_tensors_padded, all_sizes):
        all_tensors.append(tensor_[: size[0]])
    return all_tensors


def main_analysis(
    # misc
    project,
    eval_splits,
    eval_freq,
    eval_pow2,
    max_eval_steps,
    load_ckpt_path,
    save_ckpt_dir,
    compile_,
    seed,
    # data
    train_tasks,
    eval_tasks,
    batch_size,
    num_workers,
    max_bfs_width,
    # optimization
    lr,
    wd,
    lr_schedule,
    max_grad_norm,
    max_steps,
    # model
    embedding_model,
    d_text,
    seq_len,
    num_blocks,
    d_model,
    num_heads,
    d_ff,
    # contrastive learning
    use_fk_contrastive=False,
    use_row_contrastive=False,
    contrastive_weight=0.1,
    contrastive_temperature=0.07,
    run_name=None,
):
    seed_everything(seed)

    ddp = "LOCAL_RANK" in os.environ
    device = "cuda"
    if ddp:
        os.environ["OMP_NUM_THREADS"] = f"{num_workers}"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    if ddp:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if rank == 0:
        run = wandb.init(project=project, name=run_name, config=locals())
        print(run.name)

    torch.multiprocessing.set_sharing_strategy("file_system")
    torch._dynamo.config.cache_size_limit = 64  # Increased from 16 to handle dynamic shapes better
    torch._dynamo.config.compiled_autograd = compile_ if ddp else False
    torch._dynamo.config.optimize_ddp = True
    torch.set_num_threads(1)

    dataset = RelationalDataset(
        tasks=[
            (db_name, table_name, target_column, "train", columns_to_drop)
            for db_name, table_name, target_column, columns_to_drop in train_tasks
        ],
        batch_size=batch_size,
        seq_len=seq_len,
        rank=rank,
        world_size=world_size,
        max_bfs_width=max_bfs_width,
        embedding_model=embedding_model,
        d_text=d_text,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True,
        in_order=True,
    )

    eval_loaders = {}
    for db_name, table_name, target_column, columns_to_drop in eval_tasks:
        for split in eval_splits:
            eval_dataset = RelationalDataset(
                tasks=[(db_name, table_name, target_column, split, columns_to_drop)],
                batch_size=batch_size,
                seq_len=seq_len,
                rank=rank,
                world_size=world_size,
                max_bfs_width=max_bfs_width,
                embedding_model=embedding_model,
                d_text=d_text,
                seed=0,
            )
            eval_dataset.sampler.shuffle_py(0)
            eval_loaders[(db_name, table_name, split)] = DataLoader(
                eval_dataset,
                batch_size=None,
                num_workers=num_workers,
                persistent_workers=num_workers > 0,  # Only enable if num_workers > 0
                pin_memory=True,
                in_order=True,
            )

    net = RelationalTransformer(
        num_blocks=num_blocks,
        d_model=d_model,
        d_text=d_text,
        num_heads=num_heads,
        d_ff=d_ff,
        use_fk_contrastive=use_fk_contrastive,
        use_row_contrastive=use_row_contrastive,
        contrastive_weight=contrastive_weight,
        contrastive_temperature=contrastive_temperature,
    )
    if load_ckpt_path is not None:
        load_ckpt_path = Path(load_ckpt_path).expanduser()
        state_dict = torch.load(load_ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)

    if rank == 0:
        param_count = sum(p.numel() for p in net.parameters())
        print(f"{param_count=:_}")

    net = net.to(device)
    net = net.to(torch.bfloat16)
    opt = optim.AdamW(
        net.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=True,
    )

    if lr_schedule:
        lrs = optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=0.2,
            anneal_strategy="linear",
        )

    if ddp:
        net = DDP(net)

    net = torch.compile(net, dynamic=True, disable=not compile_)

    steps = 0
    if rank == 0:
        wandb.log({"epochs": 0}, step=steps)

    eval_loader_iters = {}
    for k, eval_loader in eval_loaders.items():
        eval_loader_iters[k] = iter(eval_loader)

    def evaluate(net):
        metrics = {"val": {}, "test": {}}
        net.eval()
        with torch.inference_mode():
            for (
                db_name,
                table_name,
                split,
            ), eval_loader_iter in eval_loader_iters.items():
                if table_name in [
                    "item-sales",
                    "user-ltv",
                    "item-ltv",
                    "post-votes",
                    "site-success",
                    "study-adverse",
                    "user-attendance",
                    "driver-position",
                    "ad-ctr",
                ]:
                    task_type = "reg"
                else:
                    task_type = "clf"

                preds = []
                labels = []
                losses = []
                eval_load_times = []
                
                # Analysis: track predictions per entity with their sampled label counts
                entity_analysis = defaultdict(list)  # node_idx -> list of (num_sampled_labels, prediction, ground_truth)
                
                eval_loader = eval_loaders[(db_name, table_name, split)]
                pbar = tqdm(
                    total=(
                        min(max_eval_steps, len(eval_loader))
                        if max_eval_steps is not None and max_eval_steps > -1
                        else len(eval_loader)
                    ),
                    desc=f"{db_name}/{table_name}/{split}",
                    disable=rank != 0,
                )

                batch_idx = 0
                while True:
                    tic = time.time()
                    try:
                        batch = next(eval_loader_iter)
                        batch_idx += 1
                    except StopIteration:
                        break
                    toc = time.time()
                    pbar.update(1)

                    eval_load_time = toc - tic
                    if rank == 0:
                        eval_load_times.append(eval_load_time)

                    true_batch_size = batch.pop("true_batch_size")
                    for k in batch:
                        batch[k] = batch[k].to(device, non_blocking=True)

                    batch["masks"][true_batch_size:, :] = False
                    batch["is_targets"][true_batch_size:, :] = False
                    batch["is_padding"][true_batch_size:, :] = True

                    loss, yhat_dict = net(batch)

                    if task_type == "clf":
                        yhat = yhat_dict["boolean"][batch["is_targets"]]
                        y = batch["boolean_values"][batch["is_targets"]].flatten()
                    elif task_type == "reg":
                        yhat = yhat_dict["number"][batch["is_targets"]]
                        y = batch["number_values"][batch["is_targets"]].flatten()

                    assert yhat.size(0) == true_batch_size
                    assert y.size(0) == true_batch_size

                    pred = yhat.flatten()

                    losses.append(loss.item())
                    preds.append(pred)
                    labels.append(y)
                    
                    # Analysis: Count how many previous labels of the SAME entity are in the sequence
                    if rank == 0:
                        is_targets_batch = batch["is_targets"][:true_batch_size]
                        node_idxs_batch = batch["node_idxs"][:true_batch_size]
                        is_task_nodes_batch = batch["is_task_nodes"][:true_batch_size]
                        
                        # Each sample in the batch has one target that we're predicting
                        for b in range(true_batch_size):
                            target_mask = is_targets_batch[b]
                            task_node_mask = is_task_nodes_batch[b]
                            
                            # Get all node_idxs that are targets (labels) in this sequence
                            target_node_idxs = node_idxs_batch[b][target_mask].cpu().numpy()
                            
                            if len(target_node_idxs) == 0:
                                continue
                            
                            # Each sample has one prediction and one label
                            target_pred = pred[b].cpu().item()
                            target_label = y[b].cpu().item()
                            
                            # Get the entity (node_idx) we're predicting for
                            # This is the entity ID whose future label we want to predict
                            entity_id = int(target_node_idxs[0])
                            
                            # Count how many times THIS SAME entity appears in task nodes
                            # Note: Each row appears as 2 nodes in the graph (row node + paired node)
                            # The target entity itself is also in task nodes, so we need to subtract it
                            
                            all_task_node_idxs = node_idxs_batch[b][task_node_mask].cpu().numpy()
                            num_in_task_nodes = np.sum(all_task_node_idxs == entity_id)
                            
                            # Convert nodes to records (divide by 2)
                            total_records = num_in_task_nodes // 2
                            
                            # Subtract 1 for the target itself (the current record we're predicting)
                            # This gives us the number of HISTORICAL records (past data)
                            num_historical_records = total_records - 1
                            
                            # Make sure it's not negative (in case target is not in task nodes somehow)
                            num_historical_records = max(0, num_historical_records)
                            
                            entity_analysis[entity_id].append((num_historical_records, target_pred, target_label))

                    if max_eval_steps is not None and max_eval_steps > -1 and batch_idx >= max_eval_steps:
                        break

                eval_loader_iters[(db_name, table_name, split)] = iter(eval_loader)

                pbar.close()
                preds = torch.cat(preds, dim=0)
                labels = torch.cat(labels, dim=0)

                if ddp:
                    # ensure the predictions and labels are gathered jointly
                    preds = all_gather_nd(preds)
                    labels = all_gather_nd(labels)
                else:
                    preds = [preds]
                    labels = [labels]

                if rank == 0:
                    loss = sum(losses) / len(losses)
                    k = f"loss/{db_name}/{table_name}/{split}"
                    avg_eval_load_time = sum(eval_load_times) / len(eval_load_times)
                    wandb.log(
                        {
                            k: loss,
                            f"avg_eval_load_time/{db_name}/{table_name}": avg_eval_load_time,
                        },
                        step=steps,
                    )

                    preds = torch.cat(preds, dim=0).float().cpu().numpy()
                    labels = torch.cat(labels, dim=0).float().cpu().numpy()

                    if task_type == "reg":
                        metric_name = "r2"
                        metric = r2_score(labels, preds)
                    elif task_type == "clf":
                        metric_name = "auc"
                        labels = [int(x > 0) for x in labels]
                        metric = roc_auc_score(labels, preds)

                    k = f"{metric_name}/{db_name}/{table_name}/{split}"
                    wandb.log({k: metric}, step=steps)
                    print(f"\nstep={steps}, \t{k}: {metric}")
                    metrics[split][(db_name, table_name)] = metric
                    
                    # Analysis: Distribution of sampled label counts and accuracy
                    print(f"\n{'='*80}")
                    print(f"Label Count Distribution Analysis for {db_name}/{table_name}/{split}")
                    print(f"{'='*80}")
                    
                    # Count distribution and accuracy by number of sampled entities
                    label_count_stats = defaultdict(lambda: {"total": 0, "correct": 0})
                    
                    # Debug: check prediction value range
                    all_pred_vals = []
                    all_true_vals = []
                    
                    for node_idx, records in entity_analysis.items():
                        for num_labels, pred_val, true_val in records:
                            label_count_stats[num_labels]["total"] += 1
                            all_pred_vals.append(pred_val)
                            all_true_vals.append(true_val)
                            
                            # Check if prediction is correct
                            if task_type == "clf":
                                # Use 0.0 as threshold for logits (before sigmoid)
                                # If values are already sigmoids (0-1), this still works reasonably
                                pred_class = 1 if pred_val > 0.0 else 0
                                true_class = 1 if true_val > 0 else 0
                                if pred_class == true_class:
                                    label_count_stats[num_labels]["correct"] += 1
                            elif task_type == "reg":
                                # For regression, we'll compute this differently later
                                # For now, just count total
                                pass
                    
                    # Debug: print prediction statistics
                    if task_type == "clf" and len(all_pred_vals) > 0:
                        print(f"\nPrediction value stats:")
                        print(f"  Min: {min(all_pred_vals):.4f}, Max: {max(all_pred_vals):.4f}")
                        print(f"  Mean: {np.mean(all_pred_vals):.4f}, Std: {np.std(all_pred_vals):.4f}")
                        print(f"Label distribution: {Counter(all_true_vals)}")
                    
                    # Print results sorted by number of sampled labels
                    if task_type == "clf":
                        print(f"\n{'Same Entity':<15} {'Count':<10} {'Percentage':<12} {'Accuracy':<12}")
                        print(f"{'-'*80}")
                    else:
                        print(f"\n{'Same Entity':<15} {'Count':<10} {'Percentage':<12}")
                        print(f"{'-'*80}")
                    
                    total_predictions = sum(stats["total"] for stats in label_count_stats.values())
                    for num_labels in sorted(label_count_stats.keys()):
                        stats = label_count_stats[num_labels]
                        count = stats["total"]
                        percentage = (count / total_predictions * 100) if total_predictions > 0 else 0
                        
                        if task_type == "clf":
                            accuracy = (stats["correct"] / count * 100) if count > 0 else 0
                            print(f"{num_labels:<15} {count:<10} {percentage:>10.2f}% {accuracy:>10.2f}%")
                        else:
                            print(f"{num_labels:<15} {count:<10} {percentage:>10.2f}%")
                    
                    print(f"\n{'='*80}")
                    print(f"Total predictions: {total_predictions}")
                    print(f"Total unique entities: {len(entity_analysis)}")
                    print(f"Average samples per entity: {total_predictions / len(entity_analysis):.2f}" if len(entity_analysis) > 0 else "N/A")
                    
                    if task_type == "clf":
                        total_correct = sum(stats["correct"] for stats in label_count_stats.values())
                        overall_accuracy = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
                        print(f"Overall accuracy: {overall_accuracy:.2f}%")
                    
                    print(f"{'='*80}\n")

        return metrics

    def checkpoint(best=False, db_name="", table_name=""):
        if rank != 0:
            return
        save_ckpt_dir_ = Path(save_ckpt_dir).expanduser()
        save_ckpt_dir_.mkdir(parents=True, exist_ok=True)
        if best:
            save_ckpt_path = f"{save_ckpt_dir_}/{db_name}_{table_name}_best.pt"
        else:
            save_ckpt_path = f"{save_ckpt_dir_}/{steps=}.pt"

        state_dict = net.module.state_dict() if ddp else net.state_dict()
        torch.save(state_dict, save_ckpt_path)
        print(f"saved checkpoint to {save_ckpt_path}")

    pbar = tqdm(
        total=max_steps,
        desc="steps",
        disable=rank != 0,
    )

    best_val_metrics = dict()
    best_test_metrics = dict()

    # If max_steps is 0, evaluate once and exit (eval-only mode)
    if max_steps == 0:
        metrics = evaluate(net)
        if rank == 0:
            print("\n" + "="*80)
            print("Evaluation metrics:")
            print("="*80)
            for split in eval_splits:
                for (db_name, table_name), metric in metrics[split].items():
                    print(f"{db_name}/{table_name}/{split}: {metric:.4f}")
            print("="*80 + "\n")
        if ddp:
            dist.destroy_process_group()
        return

    while steps < max_steps:
        loader.dataset.sampler.shuffle_py(int(steps / len(loader)))
        loader_iter = iter(loader)
        while steps < max_steps:
            if (eval_freq is not None and steps % eval_freq == 0) or (
                eval_pow2 and steps & (steps - 1) == 0
            ):
                metrics = evaluate(net)
                if save_ckpt_dir is not None:
                    for (db_name, table_name), metric in metrics["val"].items():
                        # Eval metric is always higher is better (auc, r2)
                        best_metric = best_val_metrics.get(
                            (db_name, table_name), -float("inf")
                        )
                        if metric > best_metric:
                            best_val_metrics[(db_name, table_name)] = metric
                            best_test_metrics[(db_name, table_name)] = metrics["test"][(db_name, table_name)]
                            checkpoint(
                                best=True,
                                db_name=db_name,
                                table_name=table_name
                            )
                        else:
                            checkpoint()

            net.train()

            tic = time.time()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            batch.pop("true_batch_size")
            for k in batch:
                batch[k] = batch[k].to(device, non_blocking=True)
            toc = time.time()
            load_time = toc - tic
            if rank == 0:
                wandb.log({"load_time": load_time}, step=steps)

            loss, _yhat_dict = net(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm = get_total_norm(
                [p.grad for p in net.parameters() if p.grad is not None]
            )
            clip_grads_with_norm_(
                net.parameters(), max_norm=max_grad_norm, total_norm=grad_norm
            )

            opt.step()
            if lr_schedule:
                lrs.step()

            steps += 1

            if ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                wandb.log(
                    {
                        "loss": loss,
                        "lr": opt.param_groups[0]["lr"],
                        "epochs": steps / len(loader),
                        "grad_norm": grad_norm,
                    },
                    step=steps,
                )

            pbar.update(1)

    # Print best test metrics
    if rank == 0:
        print("\n" + "="*80)
        print("Best test metrics:")
        print("="*80)
        for (db_name, table_name), metric in best_test_metrics.items():
            print(f"{db_name}/{table_name}/test: {metric:.4f}")
        print("="*80 + "\n")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import strictfire

    strictfire.StrictFire(main_analysis)
