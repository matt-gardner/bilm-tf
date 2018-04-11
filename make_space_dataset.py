from typing import List
import random

wsj_template = 'wsj/original/stan3-4-dependencies/wsj.%s.conllx'
outdir = 'wsj_spaces/'
num_incorrect_target_sequences = 3
max_sentence_length = 40
space_probability = 0.2  # average word length assumed to be 5

def main():
    for split in ['dev', 'test', 'train']:
        print(split)
        with open('%s/%s.tsv' % (outdir, split), 'w') as outfile:
            sentences = tokens_from_conll_file(wsj_template % split)
            for sentence in sentences:
                source_sequence = ''.join(sentence)
                correct_target_sequence = ' '.join(sentence)
                targets = [correct_target_sequence]
                for _ in range(num_incorrect_target_sequences):
                    targets.append(randomly_add_spaces(source_sequence))
                outfile.write('%s\t%s\n' % (source_sequence, '\t'.join(targets)))


def randomly_add_spaces(sentence):
    new_sentence = ''
    for c in sentence:
        new_sentence += c
        if random.random() < space_probability:
            new_sentence += ' '
    return new_sentence.strip()


def truncate_tokens(tokens: List[str]) -> List[str]:
    current_length = 0
    index = 0
    kept_tokens = []
    while current_length < max_sentence_length and index < len(tokens):
        kept_tokens.append(tokens[index])
        current_length += len(tokens[index])
        index += 1
    return kept_tokens


def tokens_from_conll_file(filename):
    sentences = []
    tokens = []
    with open(filename) as f:
        for line in f:
            fields = line.split()
            if not fields:
                sentences.append(truncate_tokens(tokens))
                tokens = []
                continue
            token = fields[1]
            tokens.append(token)
    return sentences


if __name__ == '__main__':
    main()
