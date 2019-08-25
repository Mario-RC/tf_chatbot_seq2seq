import sys, os, math, time, argparse, shutil, gzip
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

from six.moves import xrange  # pylint: disable=redefined-builtin
from datetime import datetime
import seq2seq_model_utils, data_utils


def setup_workpath(workspace):
  for p in ['data', 'nn_models', 'results', 'tf_board']:
    wp = "%s/%s" % (workspace, p)
    if not os.path.exists(wp): os.mkdir(wp)

  data_dir = "%s/data" % (workspace)
  # training data
  if not os.path.exists("%s/chat.in" % data_dir):
    n = 0
    f_zip   = gzip.open("%s/train/chat.txt.gz" % data_dir, 'rt')
    f_train = open("%s/chat.in" % data_dir, 'w')
    f_dev   = open("%s/chat_test.in" % data_dir, 'w')
    for line in f_zip:
      f_train.write(line)
      if n < 10000: 
        f_dev.write(line)
        n += 1


def exitCallback(Step, maxStep, LR, minLR):
  if (Step > maxStep and maxStep):
    raise SystemExit
    sys.exit(0)
  if (LR < minLR and minLR):
    raise SystemExit
    sys.exit(0)


def train(args):

    print("[%s] Preparing dialog data in %s" % (args.model_name, args.data_dir))
    setup_workpath(workspace=args.workspace)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(args.data_dir, args.vocab_size)

    """ tensorboard"""
    tf.reset_default_graph() # To clear the defined variables and operations of the previous cell

    total_time, print_step, step_time2, perplexity = 0, 0, 0, 0

    add_step_loss = tf.placeholder(tf.float32, name='Step_loss_ph')
    step_loss_summary = tf.summary.scalar('Params/Step_loss', add_step_loss)
    
    add_perplexity = tf.placeholder(tf.float32, name='Perplexity_ph')
    perplexity_summary = tf.summary.scalar('Params/Perplexity', add_perplexity)
    
    add_lr = tf.placeholder(tf.float32, name='Learning_rate_ph')
    lr_summary = tf.summary.scalar('Params/Learning_rate', add_lr)

    add_bucket_one_loss = tf.placeholder(tf.float32, name='Bucket1_loss_ph')
    bucket_one_loss_summary = tf.summary.scalar('Buckets/Bucket1_loss', add_bucket_one_loss)
    add_bucket_two_loss = tf.placeholder(tf.float32, name='Bucket2_loss_ph')
    bucket_two_loss_summary = tf.summary.scalar('Buckets/Bucket2_loss', add_bucket_two_loss)
    add_bucket_three_loss = tf.placeholder(tf.float32, name='Bucket3_loss_ph')
    bucket_three_loss_summary = tf.summary.scalar('Buckets/Bucket3_loss', add_bucket_three_loss)
    add_bucket_four_loss = tf.placeholder(tf.float32, name='Bucket4_loss_ph')
    bucket_four_loss_summary = tf.summary.scalar('Buckets/Bucket4_loss', add_bucket_four_loss)

    add_bucket_one_ppx = tf.placeholder(tf.float32, name='Bucket1_ppx_ph')
    bucket_one_ppx_summary = tf.summary.scalar('Buckets/Bucket1_ppx', add_bucket_one_ppx)
    add_bucket_two_ppx = tf.placeholder(tf.float32, name='Bucket2_ppx_ph')
    bucket_two_ppx_summary = tf.summary.scalar('Buckets/Bucket2_ppx', add_bucket_two_ppx)
    add_bucket_three_ppx = tf.placeholder(tf.float32, name='Bucket3_ppx_ph')
    bucket_three_ppx_summary = tf.summary.scalar('Buckets/Bucket3_ppx', add_bucket_three_ppx)
    add_bucket_four_ppx = tf.placeholder(tf.float32, name='Bucket4_ppx_ph')
    bucket_four_ppx_summary = tf.summary.scalar('Buckets/Bucket4_ppx', add_bucket_four_ppx)

    #total_loss = tf.Variable(0.0)
    #with tf.name_scope('Total_loss'):
    #  tot_loss = total_loss.assign_add(add_step_loss)
    #tf.summary.scalar('Total_loss', tot_loss)

    # For tf-gpu
    configgpu = tf.ConfigProto(allow_soft_placement=True) #allow_soft_placement=True, log_device_placement=True
    configgpu.gpu_options.allocator_type ='BFC'
    configgpu.gpu_options.allow_growth = True
    configgpu.gpu_options.per_process_gpu_memory_fraction = args.gpu_usage

    with tf.Session(config=configgpu) as sess:

        # Create model.
        print("Creating %d layers of %d units." % (args.num_layers, args.size))
        model = seq2seq_model_utils.create_model(sess, args, forward_only=False)

        # TensorBoard
        writer = tf.summary.FileWriter(args.tf_board_dir, graph=sess.graph)
        #sess.run(tf.global_variables_initializer())
        #ckpt = tf.train.get_checkpoint_state(args.model_dir)
        #if ckpt and ckpt.model_checkpoint_path: # if exists
          #writer = tf.summary.FileWriterCache.get(args.tf_board_dir)
          #writer.reopen()
        #else:
          #writer = tf.summary.FileWriter(args.tf_board_dir, graph=sess.graph)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)." % args.max_train_data_size)
        dev_set = data_utils.read_data(dev_data, args.buckets, reversed=args.rev_model)
        train_set = data_utils.read_data(train_data, args.buckets, args.max_train_data_size, reversed=args.rev_model)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(args.buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        loss = 0.0
        step_time, step_time_verbosity, step_time_checkpoint = 0.0, 0.0, 0.0
        current_step = 0
        previous_losses = []
        bucket_loss = [0,0,0,0]
        bucket_ppx = [0,0,0,0]

        # Load vocabularies.
        vocab_path = os.path.join(args.data_dir, "vocab%d.in" % args.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        while True:
          tf.logging.set_verbosity(tf.logging.INFO)

          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              train_set, bucket_id)
          
          # print("[shape]", np.shape(encoder_inputs), np.shape(decoder_inputs), np.shape(target_weights))
          _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                         target_weights, bucket_id, forward_only=False, force_dec_input=True)

          loss += step_loss / args.steps_per_checkpoint
          
          #step_time = (time.time() - start_time) / args.steps_per_checkpoint
          step_time = (time.time() - start_time)
          step_time_verbosity += (step_time / args.steps_per_verbosity)
          step_time_checkpoint += (step_time / args.steps_per_checkpoint)
          current_step += 1
          
          # Once in a while we print statistics.
          if (current_step % args.steps_per_verbosity == 0):
            print("    loss: %.4f step: %d time: %.2f" % (step_loss, model.global_step.eval(), step_time_verbosity))
            step_verbosity, step_time_verbosity = 0.0, 0.0
          
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if (current_step % args.steps_per_checkpoint == 0):
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("\nglobal step %d learning rate %.4f step-time %.2f perplexity %.2f @ %s" %
                   (model.global_step.eval(), model.learning_rate.eval(), step_time_checkpoint, perplexity, datetime.now()))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time_checkpoint, loss = 0.0, 0.0

            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(args.buckets)):
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(dev_set, bucket_id)
              _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, 
                                          target_weights, bucket_id, forward_only=True, force_dec_input=False)

              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))

              bucket_loss[bucket_id] = eval_loss
              bucket_ppx[bucket_id] = eval_ppx
            
            sys.stdout.flush()
          
          # Once in a while, we save summary tensorboard and print statistics.
          if (current_step % args.steps_per_checkpoint == 0):
            merge = tf.summary.merge_all()
            summary = sess.run(merge, feed_dict={
            add_step_loss: step_loss, add_perplexity: perplexity, add_lr: model.learning_rate.eval(),
            add_bucket_one_loss: bucket_loss[0],   add_bucket_one_ppx: bucket_ppx[0],
            add_bucket_two_loss: bucket_loss[1],   add_bucket_two_ppx: bucket_ppx[1],
            add_bucket_three_loss: bucket_loss[2], add_bucket_three_ppx: bucket_ppx[2],
            add_bucket_four_loss: bucket_loss[3],  add_bucket_four_ppx: bucket_ppx[3]
            })
            writer.add_summary(summary, model.global_step.eval()) # Record for tensorboard
            #writer.flush()
          if (current_step % args.steps_per_summary == 0):
            summary_train_loss, val_loss = sess.run([step_loss_summary, add_step_loss], feed_dict={add_step_loss: step_loss})
            writer.add_summary(summary_train_loss, model.global_step.eval()) # Record for tensorboard
            summary_train_lr, val_lr = sess.run([lr_summary, add_lr], feed_dict={add_lr: model.learning_rate.eval()})
            writer.add_summary(summary_train_lr, model.global_step.eval()) # Record for tensorboard
            #writer.flush()
            
          exitCallback(model.global_step.eval(),args.max_train_steps,
                       model.learning_rate.eval(),args.min_train_lr)
