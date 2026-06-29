import argparse
from pathlib import Path

from simulator.controllers import QLearningController
from simulator.experiment import evaluate_baseline, evaluate_controller


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate smart-room controllers.")
    parser.add_argument("--policy", default="q_table.json", help="Path to a learned Q-table.")
    parser.add_argument("--days", type=int, default=7, help="Number of simulated days to evaluate.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for the evaluation scenario.")
    args = parser.parse_args()

    baseline = evaluate_baseline(days=args.days, seed=args.seed)
    print("Baseline controller:")
    for key, value in baseline.as_dict().items():
        print(f"  {key}: {value}")

    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"\nNo learned policy found at {policy_path}. Run `python main.py` first.")
        return

    learner = QLearningController(epsilon=0.0)
    learner.load(policy_path)
    learned = evaluate_controller(learner, days=args.days, seed=args.seed)

    print("\nLearned controller:")
    for key, value in learned.as_dict().items():
        print(f"  {key}: {value}")

    print("\nDelta versus baseline:")
    print(f"  reward: {round(learned.total_reward - baseline.total_reward, 3)}")
    print(f"  energy_wh: {round(learned.total_energy_wh - baseline.total_energy_wh, 3)}")
    print(
        "  comfort_when_occupied: "
        f"{round(learned.average_comfort_when_occupied - baseline.average_comfort_when_occupied, 3)}"
    )


if __name__ == "__main__":
    main()
