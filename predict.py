import os
from shutil import copyfile

import tensorflow as tf
from datetime import datetime

import data_utils
from seq2seq_model_utils import create_model, get_predicted_sentence, cal_bleu


def predict(args, debug=False):
    def _get_test_dataset():
        with open(args.test_dataset_path) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results', str(args.num_layers), str(args.size), str(args.beam_size)])
    results_path = os.path.join(args.results_dir, results_filename+'.txt')
    if os.path.isfile(results_path):
        for i in range(1,10):
            results_path = os.path.join(args.results_dir, results_filename+'({}).txt'.format(i))
            if not os.path.isfile(results_path):
                break

    with tf.Session() as sess, open(results_path, 'w') as results_fh:
        # Create model and load parameters.
        args.batch_size = 1
        model = create_model(sess, args)

        # Load vocabularies.
        vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        test_dataset = _get_test_dataset()

        for sentence in test_dataset:
            # Get token-ids for the input sentence.
            predicted_sentence = get_predicted_sentence(args, sentence, vocab, rev_vocab, model, sess, debug=debug)
            if isinstance(predicted_sentence, list):
                print("%s : (%s)" % (sentence, datetime.now()))
                results_fh.write("%s : (%s)\n" % (sentence, datetime.now()))
                for sent in predicted_sentence:
                    print("  (%.4f) -> %s" % (sent['prob'], sent['dec_inp']))
                    results_fh.write("  (%.4f) -> %s\n" % (sent['prob'], sent['dec_inp']))
            else:
                print(sentence, ' -> ', predicted_sentence)
                results_fh.write("%s -> %s\n" % (sentence, predicted_sentence))
            # break

    results_fh.close()
    print("results written in %s" % results_path)
