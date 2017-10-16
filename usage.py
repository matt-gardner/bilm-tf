import json
import tensorflow as tf
import os
import tqdm
import numpy
import h5py
from bilm import Batcher, BidirectionalLanguageModel

class Token:
    def __init__(self, text, pos_tag, dep_label, prev_pos, prev_dep):
        self.text = text
        self.pos_tag = pos_tag
        self.dep_label = dep_label
        self.prev_pos = prev_pos
        self.prev_dep = prev_dep
        self.next_pos = 'NONE'
        self.next_dep = 'NONE'


def main():
    # Location of pretrained LM.  Here we use the test fixtures.
    vocab_file = 'vocab.txt'
    options_file = 'skip_connections_options.json'
    weight_file = 'fine_tuned_weights.hdf5'

    # Create a Batcher to map text to character ids.
    batcher = Batcher(vocab_file, 50)

    with tf.device('/gpu:0'):
        # Input placeholders to the biLM.
        context_character_ids = tf.placeholder('int32', shape=(None, None, 50))

        # Build the graphs.  It is necessary to specify a reuse
        # variable scope (with name='') for every placeholder except the first.
        # All variables names for the bidirectional LMs are prefixed with 'bilm/'.
        context_model = BidirectionalLanguageModel(
            options_file, weight_file, context_character_ids)

        # Get ops to compute the LM embeddings.
        context_embeddings_op = context_model.get_ops()['lm_embeddings']

    train_file = 'wsj/original/stan3-4-dependencies/wsj.train.conllx'
    dev_file = 'wsj/original/stan3-4-dependencies/wsj.dev.conllx'
    test_file = 'wsj/original/stan3-4-dependencies/wsj.test.conllx'

    print("Converting dev data")
    convert_file(batcher, context_character_ids, context_embeddings_op, dev_file, 'data_skip_ft/dev/')
    print("Converting training data")
    convert_file(batcher, context_character_ids, context_embeddings_op, train_file, 'data_skip_ft/train/')
    print("Converting test data")
    convert_file(batcher, context_character_ids, context_embeddings_op, test_file, 'data_skip_ft/test/')


def tokens_from_conll_file(filename):
    prev_pos = 'NONE'
    prev_dep = 'NONE'
    sentences = []
    tokens = []
    last_token = None
    with open(filename) as f:
        for line in f:
            fields = line.split()
            if not fields:
                sentences.append(tokens)
                tokens = []
                last_token = None
                continue
            text = fields[1]
            pos_tag = fields[3]
            dep_label = fields[7]
            token = Token(text, pos_tag, dep_label, prev_pos, prev_dep)
            if last_token:
                last_token.next_pos = pos_tag
                last_token.next_dep = dep_label
            prev_pos = pos_tag
            prev_dep = dep_label
            tokens.append(token)
            last_token = token
    return sentences

def group_by_count(iterable, count):
    return list(zip(*[iterable[i::count] for i in range(count)]))


def convert_file(batcher, context_character_ids, context_embeddings_op, token_file, out_dir):
    # Now we can compute embeddings.
    sentences = tokens_from_conll_file(token_file)
    grouped_sentences = group_by_count(sentences, 100)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # It is necessary to initialize variables once before running inference.
        sess.run(tf.global_variables_initializer())

        for batch_num, sentence_group in tqdm.tqdm(list(enumerate(grouped_sentences))):
            sentence_text = [[token.text for token in sentence] for sentence in sentence_group]
            # Create batches of data.
            context_ids = batcher.batch_sentences(sentence_text)

            # Compute LM embeddings.
            context_embeddings = sess.run(
                [context_embeddings_op],
                feed_dict={context_character_ids: context_ids}
            )

            token_jsons = []
            token_embeddings = []
            for sentence_tokens, sentence_embeddings in zip(sentence_group, context_embeddings[0]):
                embedding = sentence_embeddings
                for i, token in enumerate(sentence_tokens):
                    token_json = {
                            'text': token.text,
                            'pos_tag': token.pos_tag,
                            'dep_label': token.dep_label,
                            'prev_pos': token.prev_pos,
                            'prev_dep': token.prev_dep,
                            'next_pos': token.next_pos,
                            'next_dep': token.next_dep,
                            }
                    token_embeddings.append([embedding[j, i, :] for j in range(3)])
                    token_jsons.append(token_json)
            token_embeddings = numpy.asarray(token_embeddings)
            with h5py.File(out_dir + 'batch_%d_embeddings.h5' % batch_num, 'w') as out:
                out.create_dataset('embeddings', data=token_embeddings)
            with open(out_dir + 'batch_%d_tokens.json' % batch_num, 'w') as out:
                json.dump(token_jsons, out, indent=2)


if __name__ == '__main__':
    main()
