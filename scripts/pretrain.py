from rt.main import main
from rt.tasks import all_tasks, forecast_tasks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="rel-f1")
args = parser.parse_args()

if __name__ == "__main__":
    run_name = f"leave_{args.dataset}_{args.seed}"
    main(
        # misc
        project="rt",
        run_name=run_name,
        eval_splits=["val", "test"],
        eval_freq=1_000,
        eval_pow2=True,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir=f"ckpts/leave_{args.dataset}_{args.seed}",
        compile_=True,
        seed=args.seed,
        # data
        train_tasks=[t for t in all_tasks if t[0] != args.dataset],
        eval_tasks=[t for t in forecast_tasks if t[0] == args.dataset],
        batch_size=32,
        num_workers=0, 
        max_bfs_width=256,
        # optimization
        lr=1e-3,
        wd=0.1,
        lr_schedule=True,
        max_grad_norm=1.0,
        max_steps=50_001,
        # model
        embedding_model="all-MiniLM-L12-v2",
        d_text=384,
        seq_len=1024,
        num_blocks=12,
        d_model=256,
        num_heads=8,
        d_ff=1024,
    )
