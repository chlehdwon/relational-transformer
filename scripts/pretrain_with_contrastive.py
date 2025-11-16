from rt.main import main
from rt.tasks import all_tasks, forecast_tasks
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="rel-f1")
parser.add_argument("--use_fk", action="store_true", help="Use FK contrastive loss")
parser.add_argument("--use_row", action="store_true", help="Use row contrastive loss")
parser.add_argument("--contrastive_weight", type=float, default=0.1)
parser.add_argument("--c_temperature", type=float, default=0.07)
args = parser.parse_args()


if __name__ == "__main__":
    tag = "_full" if args.use_fk and args.use_row else "_fk" if args.use_fk else "_row" if args.use_row else ""
    contrast_type = "full" if args.use_fk and args.use_row else "fk" if args.use_fk else "row" if args.use_row else "none"
    run_name = f"leave_{args.dataset}_{contrast_type}_contrast_{args.seed}"
    main(
        # misc
        project="rt",
        run_name=run_name,
        eval_splits=["val", "test"],
        eval_freq=1_000,
        eval_pow2=True,
        max_eval_steps=40,
        load_ckpt_path=None,
        save_ckpt_dir=f"ckpts/leave_{args.dataset}_contrastive{tag}_{args.contrastive_weight:.2f}_{args.c_temperature:.2f}_{args.seed}",
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
        # contrastive learning
        use_fk_contrastive=args.use_fk,
        use_row_contrastive=args.use_row,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.c_temperature,
    )

