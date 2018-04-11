import json
import h5py
import numpy
import tqdm

from keras.models import Model
from keras.layers import Dense, Input

def main():
    dev_batches = 17
    train_batches = 390
    test_batches = 24
    num_epochs = 10

    experiments = [
            ('data/pos_vocab.txt', 'pos_tag'),
            ('data/dep_vocab.txt', 'dep_label'),
            ('data/pos_vocab.txt', 'prev_pos'),
            ('data/pos_vocab.txt', 'next_pos'),
            ('data/dep_vocab.txt', 'prev_dep'),
            ('data/dep_vocab.txt', 'next_dep'),
            ]


    print("Building model")

    results = {}
    for vocab_file, label_field in experiments:
        weight_file = 'results/skip_ft/%s_weights.h5' % label_field
        data_dir = 'data_skip_ft/'
        vector_dim = 1024
        token_to_index, index_to_token = read_vocab(vocab_file)
        print("Running experiment for", label_field)
        max_accuracies = []
        test_accuracies = []
        for embedding_layer in range(4):
            model = build_model(vector_dim, len(token_to_index))
            print('Running embedding layer', embedding_layer)
            accuracies = []
            for epoch in range(num_epochs):
                model.fit_generator(batch_generator(data_dir + 'train/',
                                                    train_batches,
                                                    label_field,
                                                    token_to_index,
                                                    embedding_layer),
                                    steps_per_epoch=train_batches,
                                    epochs=1)
                result = model.evaluate_generator(batch_generator(data_dir + 'dev/',
                                                                  dev_batches,
                                                                  label_field,
                                                                  token_to_index,
                                                                  embedding_layer),
                                                  steps=dev_batches)
                accuracy = result[1]
                accuracies.append(accuracy)
                if accuracy == max(accuracies):
                    model.save_weights(weight_file)
            max_accuracies.append(max(accuracies))
            model.load_weights(weight_file)
            result = model.evaluate_generator(batch_generator(data_dir + 'test/',
                                                              test_batches,
                                                              label_field,
                                                              token_to_index,
                                                              embedding_layer),
                                              steps=test_batches)
            test_accuracies.append(result[1])
        print(label_field, max_accuracies, test_accuracies)
        results[label_field] = (max_accuracies, test_accuracies)
    for label_field, (max_accuracies, test_accuracies) in results.items():
        print('%s:' % label_field)
        print('  Dev:', max_accuracies)
        print('  Test:', test_accuracies)


def read_vocab(filename):
    with open(filename) as f:
        items = [line.strip() for line in f]
    token_to_index = {}
    index_to_token = {}
    for token in items:
        index = len(token_to_index)
        token_to_index[token] = index
        index_to_token[index] = token
    return token_to_index, index_to_token


def batch_generator(directory: str, num_batches: int, field: str, token_to_index, embedding_layer):
    while True:
        for batch_num in range(num_batches):
            tokens, embeddings = read_batch(directory, batch_num)
            labels = tokens_to_labels(tokens, field, token_to_index)
            inputs = embeddings_to_inputs(embeddings, embedding_layer)
            yield inputs, labels


def read_batch(directory: str, batch_num: int):
    with open(directory + 'batch_%d_tokens.json' % batch_num) as f:
        tokens = json.load(f)
    with h5py.File(directory + 'batch_%d_embeddings.h5' % batch_num, 'r') as h5f:
        embeddings = h5f['embeddings'][:]
    return tokens, embeddings


def tokens_to_labels(tokens, field, token_to_index):
    target_indices = numpy.asarray([token_to_index[token[field]] for token in tokens])
    targets = numpy.zeros((len(tokens), len(token_to_index)))
    targets[numpy.arange(len(tokens)), target_indices] = 1
    return targets


def embeddings_to_inputs(embeddings, layer: int):
    if layer == 3:
        return numpy.mean(embeddings, axis=1)
    else:
        return embeddings[:, layer, :]


def build_model(input_dim: int, num_labels: int):
    input_layer = Input(shape=(input_dim,))
    predictions = Dense(num_labels, activation='softmax')(input_layer)
    model = Model(inputs=input_layer, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


if __name__ == '__main__':
    main()
