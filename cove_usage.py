
from spliced.cove import CoVe
from dl.data import LMTaskVocabulary
from keras.layers import Input
from keras.models import Model

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


sentences = [
    ['The', 'first', 'OOV_TOKEN', '.'], ['Second', 'sentence', '.']
]

X_token_ids = vocab.batch_sentences_to_X(sentences)['tokens']
cove_embeddings = cove_model.predict_on_batch(X_token_ids)
