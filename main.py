import argparse

from src.predict import predict_image
from src.realtime import run_realtime
from src.train import train_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emotion Detection from Face")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", choices=["knn", "ann"], default="knn")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)

    predict_parser = subparsers.add_parser("predict", help="Predict emotion for one image")
    predict_parser.add_argument("--image", required=True, help="Path to image")

    realtime_parser = subparsers.add_parser("realtime", help="Run webcam emotion detection")
    realtime_parser.add_argument("--camera", type=int, default=0, help="Camera index")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_and_evaluate(model_type=args.model, test_size=args.test_size, random_state=args.random_state)
    elif args.command == "predict":
        label, confidence = predict_image(args.image)
        print(f"Predicted Emotion: {label}")
        print(f"Confidence: {confidence:.4f}")
    elif args.command == "realtime":
        run_realtime(camera_index=args.camera)


if __name__ == "__main__":
    main()
