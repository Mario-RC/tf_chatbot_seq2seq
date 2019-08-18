import os, sys, argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import logging

from config import params_setup
from train import train
from predict import predict
from chat import chat


def main(_):
    args = params_setup()
    print("[args]: ", args)
    if args.mode == 'train':
      train(args)
    elif args.mode == 'test':
      predict(args)
    elif args.mode == 'chat':
      chat(args)


if __name__ == "__main__":
    tf.logging.set_verbosity(logging.INFO)
    tf.app.run()