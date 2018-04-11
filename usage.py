import json
import tensorflow as tf
import os
import tqdm
import numpy
import h5py
from bilm import Batcher, BidirectionalLanguageModel
from spliced.cove import CoVe
from dl.data import LMTaskVocabulary
from keras.layers import Input
from keras.models import Model

class Token:
    def __init__(self, text, pos_tag, dep_label, prev_pos, prev_dep):
        self.text = text
        self.pos_tag = pos_tag
        self.dep_label = dep_label
        self.prev_pos = prev_pos
        self.prev_dep = prev_dep
        self.next_pos = 'NONE'
        self.next_dep = 'NONE'

COVE = True
vocab_file = 'vocab.txt'
options_file = 'skip_connections_options.json'
weight_file = 'fine_tuned_weights.hdf5'
out_dir = 'data_cove/'

def main():

    if COVE:
        vocab = LMTaskVocabulary(
            'ptb_unique_tokens_lower.txt', None, None,
            lm_normalizers=[lambda x: x.lower()],
            use_char_inputs=False,
            mask_zero=True,
            use_token_key=False,
            add_bos_eos=False
        )

        cove_encoder = CoVe(
            {'embedding_weight_file': 'ptb_unique_tokens_lower_glove300.hdf5',
             'cove_weight_file': 'cove_parameters.hdf5'}
        )
        token_ids = Input((None, ), dtype='int32')
        cove_layers = cove_encoder(token_ids, return_layers=True)
        cove_model = Model(inputs=token_ids, outputs=cove_layers)
        cove_model.compile(loss='mse', optimizer='sgd')

        # UGLY HACK!
        batcher = vocab
        context_character_ids = None
        context_embeddings_op = cove_model
    else:
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
    convert_file(batcher, context_character_ids, context_embeddings_op, dev_file, out_dir + 'dev/')
    print("Converting training data")
    convert_file(batcher, context_character_ids, context_embeddings_op, train_file, out_dir + 'train/')
    print("Converting test data")
    convert_file(batcher, context_character_ids, context_embeddings_op, test_file, out_dir + 'test/')


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
    if COVE:
        vocab = batcher
        cove_model = context_embeddings_op
        for batch_num, sentence_group in tqdm.tqdm(list(enumerate(grouped_sentences))):
            sentence_text = [[token.text for token in sentence] for sentence in sentence_group]
            X_token_ids = vocab.batch_sentences_to_X(sentence_text)['tokens']
            cove_embeddings = cove_model.predict_on_batch(X_token_ids)
            # Duplicate the first layer, so that we have a uniformly-shaped tensor.
            cove_embeddings[0] = numpy.concatenate([cove_embeddings[0], cove_embeddings[0]], axis=-1)
            # This gives us the same shape result as we have with the non-CoVe case.
            context_embeddings = [numpy.stack(cove_embeddings, axis=1)]
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
                    token_embeddings.append([embedding[j][i, :] for j in range(3)])
                    token_jsons.append(token_json)
            token_embeddings = numpy.asarray(token_embeddings)
            with h5py.File(out_dir + 'batch_%d_embeddings.h5' % batch_num, 'w') as out:
                out.create_dataset('embeddings', data=token_embeddings)
            with open(out_dir + 'batch_%d_tokens.json' % batch_num, 'w') as out:
                json.dump(token_jsons, out, indent=2)
    else:
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
                        token_embeddings.append([embedding[j][i, :] for j in range(3)])
                        token_jsons.append(token_json)
                token_embeddings = numpy.asarray(token_embeddings)
                with h5py.File(out_dir + 'batch_%d_embeddings.h5' % batch_num, 'w') as out:
                    out.create_dataset('embeddings', data=token_embeddings)
                with open(out_dir + 'batch_%d_tokens.json' % batch_num, 'w') as out:
                    json.dump(token_jsons, out, indent=2)


if __name__ == '__main__':
    main()
