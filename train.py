"""Train all sentiment models and print a summary table."""
from src.sentiment.models import train_all_models, MODEL_DISPLAY


def main() -> None:
    print("Training sentiment models on UCI Sentiment Labelled Sentences...\n")
    results = train_all_models()

    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Train(s)':>10}"
    print(header)
    print("-" * len(header))
    for key in MODEL_DISPLAY:
        m = results[key]
        print(
            f"{m.name:<25} {m.accuracy:>10.4f} {m.precision:>10.4f} "
            f"{m.recall:>10.4f} {m.f1:>10.4f} {m.train_seconds:>10.2f}"
        )
    print("\nArtefacts saved to ./models/")


if __name__ == "__main__":
    main()
