import argparse
from simulator.experiment import evaluate_controller, train_q_controller
def main() -> None:
    parser = argparse.ArgumentParser(description="Train the smart-room adaptive controller.")
    parser.add_argument("--days", type=int, default=45, help="Number of simulated days to train.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible experiments.")
    parser.add_argument("--output", default="q_table.json", help="Where to save the learned policy.")
    args = parser.parse_args()
    controller = train_q_controller(days=args.days, seed=args.seed, save_path=args.output)
    summary = evaluate_controller(controller, days=7, seed=args.seed + 1)
    print("Training complete")
    print(f"Saved policy: {args.output}")
    print(f"Q entries: {len(controller.q)}")
    print("Seven-day validation:")
    for key, value in summary.as_dict().items():
        print(f"  {key}: {value}")
if __name__ == "__main__":
    main()
