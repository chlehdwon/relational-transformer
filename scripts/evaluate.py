from rt.main_analysis import main_analysis
from rt.tasks import all_tasks, forecast_tasks, forecast_tasks_dict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--task", type=str, default="driver-top3")
# parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file for evaluation")
args = parser.parse_args()

if __name__ == "__main__":
    run_name = f"eval_analysis_{args.dataset}_{args.task}_{args.seed}"
    args.checkpoint = f"/data/scratch/rt_ckpts/contd-pretrain_{args.dataset}_{args.task}.pt"
    main_analysis(
        # misc
        project="rt",
        run_name=run_name,
        eval_splits=["test"],  # Only evaluate on test split
        eval_freq=0,  # Evaluate only at initialization (no training)
        eval_pow2=False,
        max_eval_steps=None,  # Evaluate on full dataset
        load_ckpt_path=args.checkpoint,
        save_ckpt_dir=None,
        compile_=True,
        seed=args.seed,
        # data
        train_tasks=[forecast_tasks_dict[args.task]],
        eval_tasks=[forecast_tasks_dict[args.task]],
        batch_size=32,
        num_workers=0,
        max_bfs_width=256,
        # optimization (not used since max_steps=0)
        lr=1e-4,
        wd=0.0,
        lr_schedule=False,
        max_grad_norm=1.0,
        max_steps=0,  # No training steps
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
    )
