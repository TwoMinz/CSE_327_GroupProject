"""
Main entry point for crowd counting project.
"""

import os
import argparse
import torch

from scripts.train import train
from scripts.evaluate import evaluate
from scripts.predict import predict
from src.config import MODEL_CONFIG, TRAIN_CONFIG, EVAL_CONFIG, DEVICE


def main():
    """
    Main entry point for crowd counting project.
    """
    parser = argparse.ArgumentParser(description='Crowd Counting with TransCrowd')
    subparsers = parser.add_subparsers(dest='mode', help='operation mode')

    # Train parser
    train_parser = subparsers.add_parser('train', help='train model')
    train_parser.add_argument('--name', type=str, default=None, help='experiment name')
    train_parser.add_argument('--resume', type=str, default=None, help='path to latest checkpoint')

    # Evaluate parser
    eval_parser = subparsers.add_parser('evaluate', help='evaluate model')
    eval_parser.add_argument('--name', type=str, default=None, help='experiment name')
    eval_parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')
    eval_parser.add_argument('--num-samples', type=int, default=10, help='number of samples to visualize')

    # Predict parser
    predict_parser = subparsers.add_parser('predict', help='predict with model')
    predict_parser.add_argument('--input', type=str, required=True, help='path to input image or directory')
    predict_parser.add_argument('--name', type=str, default=None, help='experiment name')
    predict_parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint')

    args = parser.parse_args()

    # Print configuration
    print(f"Using device: {DEVICE}")
    print(f"Model configuration: {MODEL_CONFIG}")

    # Run selected mode
    if args.mode == 'train':
        print(f"Training configuration: {TRAIN_CONFIG}")
        train(args)
    elif args.mode == 'evaluate':
        print(f"Evaluation configuration: {EVAL_CONFIG}")
        evaluate(args)
    elif args.mode == 'predict':
        predict(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()