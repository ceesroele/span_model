"""
SpanModel
  - Based on a text, generates spans within that text and their classification.

Uses the dataclass `Fragment` which contains (start_index, end_index, label) of
a fragment, where start_index is inclusive and end_index is exclusive.

For a more extensive explanation with examples, see:
https://propaganda.math.unipd.it/semeval2021task6/index.html

The current SpanModel was originally developed as a solution to subtask 2 of
SemEval-2021 Task 6: "Detection of Persuasion Techniques in Texts and Images".
"""
from simpletransformers.seq2seq import Seq2SeqModel, Seq2SeqArgs

import logging
import sys
from pprint import pprint

from .span_utils import (
    get_tokens_from_labels,
    span_data_to_dataframe,
    decode,
    calibrate,
    split_in_sentences,
    merge_short_sentences
)

# Use this logger in all modules
logger = logging.getLogger("span")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class SpanModel(Seq2SeqModel):
    """SpanModel is an extension of Simple Transformers Seq2SeqModel,
    please see its documentation for more info.

    Added argument:
      labels: required field with the list of all labels used for classification
      split_in_sentences: if True, a text for which spans are to be generated is first split into
            sentences and the results after which the results per sentence are merged.
    """
    def __init__(
            self,
            encoder_type=None,
            encoder_name=None,
            decoder_name=None,
            encoder_decoder_type=None,
            encoder_decoder_name=None,
            labels=None,
            split_in_sentences=False,
            additional_special_tokens_encoder=None,
            additional_special_tokens_decoder=None,
            config=None,
            args=None,
            use_cuda=True,
            cuda_device=-1,
            **kwargs,
    ):
        super().__init__(encoder_type=encoder_type,
                         encoder_name=encoder_name,
                         encoder_decoder_type=encoder_decoder_type,
                         encoder_decoder_name=encoder_decoder_name,
                         additional_special_tokens_encoder=additional_special_tokens_encoder,
                         additional_special_tokens_decoder=additional_special_tokens_decoder,
                         config=config,
                         args=args,
                         use_cuda=use_cuda,
                         cuda_device=cuda_device,
                         **kwargs)

        assert labels is not None, "Missing value for required field: labels"
        self.labels = labels
        self.split_in_sentences = split_in_sentences
        tokens = get_tokens_from_labels(labels)

        self.encoder_tokenizer.add_tokens(tokens)
        len_encoder_tokenizer = len(self.encoder_tokenizer)
        if encoder_decoder_type == 'bart':
            self.model.resize_token_embeddings(len_encoder_tokenizer)
        else:
            self.model.encoder.resize_token_embeddings(len_encoder_tokenizer)

        self.decoder_tokenizer.add_tokens(tokens)
        len_decoder_tokenizer = len(self.decoder_tokenizer)
        if encoder_decoder_type == 'bart':
            self.model.resize_token_embeddings(len_decoder_tokenizer)
        else:
            self.model.decoder.resize_token_embeddings(len_decoder_tokenizer)


    def train_model(self, train_data, output_dir=None, show_running_loss=True, random_mask=0.0,
        args=None, eval_data=None, verbose=True, **kwargs):
        """
        args:
            train_data: list of dictionaries with keys ['id', 'article', 'fragments'],
               where 'fragments' is a list of `Fragment` dataclass objects.
            eval_data: same structure as train_data
        """
        pprint(args)
        if random_mask > 0.0:
            import copy
            tmp_args = copy.deepcopy(args)
            # First we prime with unmasked data
            tmp_args['num_train_epochs'] = 2
            logger.info("Train 2 episodes without mask")
            train_data_df = span_data_to_dataframe(train_data, self.labels, random_mask=0.0)
            eval_data_df = span_data_to_dataframe(eval_data, self.labels, random_mask=0.0)
            super().train_model(train_data_df, output_dir=output_dir, show_running_loss=show_running_loss,
                                args=tmp_args, eval_data=eval_data_df, verbose=verbose, **kwargs)

            logger.info("Train 3 * 5 episodes with different masks")
            for _ in range(3):
                train_data_df = span_data_to_dataframe(train_data, self.labels, random_mask=random_mask)
                eval_data_df = span_data_to_dataframe(eval_data, self.labels, random_mask=random_mask)
                tmp_args['num_train_epochs'] = 5
                super().train_model(train_data_df, output_dir=output_dir,
                                    show_running_loss=show_running_loss,
                                    args=tmp_args, eval_data=eval_data_df, verbose=verbose, **kwargs)

        #///logger.info(f"Regular training for {args['num_train_epochs']}")
        train_data_df = span_data_to_dataframe(train_data, self.labels, random_mask=0.0)
        eval_data_df = span_data_to_dataframe(eval_data, self.labels, random_mask=0.0)
        super().train_model(train_data_df, output_dir=output_dir, show_running_loss=show_running_loss,
                            args=args, eval_data=eval_data_df, verbose=verbose, **kwargs)


    def eval_model(self, eval_data, output_dir=None, verbose=True, silent=False, **kwargs):
        eval_data_df = span_data_to_dataframe(eval_data, self.labels)
        return super().eval_model(eval_data_df, output_dir=output_dir, verbose=verbose,
                                  silent=silent, **kwargs)


    def predict(self, to_predict):
        predictions = super().predict(to_predict)
        if self.split_in_sentences:
            preds = self._interpret_outcome_split_in_sentences(to_predict)
        else:
            preds = self._interpret_outcome(to_predict, predictions)
        return preds

    def _interpret_outcome(self, to_predict, predictions):
        x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r']
        res = []
        for s_index in range(len(to_predict)):
            if type(predictions[0]) == list:
                print("====== getting returns as list")
                top_article = None
                top_fragments = None
                min_errors = 1000
                for i in range(len(predictions[0])):
                    # original
                    article, fragments, errors = decode(predictions[s_index][i], self.labels)

                    # orig_text = data[s_index]['text']

                    # BEGIN alternative
                    # lst = insert_tags_list(orig_text, out_sentences[s_index][i])
                    # tagged_sentence = insert_tags(orig_text, lst)
                    # article, fragments, errors = decode(tagged_sentence, all_labels)
                    # END alternative

                    # print(f"orig ({len(orig_text)}: {orig_text}")
                    # for j, f in enumerate(fragments):
                    #    print(f"{x[j]}: {f.extract(orig_text)}")
                    # print(f"generated ({len(article)}): {article}")
                    # for j, f in enumerate(fragments):
                    #    print(f"{x[j]}: [{f.label}] {f.extract(article)}")

                    if errors < min_errors:
                        top_article = article
                        top_fragments = fragments
                        min_errors = errors
                article = top_article
                fragments = top_fragments
            else:
                # tokenized_list = insert_tags_list(sentences[s_index], out_sentences[s_index])
                # tagged_sentence = insert_tags(sentences[s_index], tokenized_list)
                # article, fragments, errors = decode(tagged_sentence, all_labels)

                # print("SENTENCES=[",sentences[s_index],"]")
                # print("OUT SENTENCE=[",out_sentences[s_index],"]")
                # print("TAGGED SENTENCE=[",tagged_sentence, "]")

                article, fragments, errors = decode(predictions[s_index], self.labels)

            for i in range(len(fragments)):
                fragments[i] = calibrate(fragments[i], article, to_predict[s_index], distance=8)

            res.append({'id': f'<unknown_{s_index}>', 'article': to_predict[s_index], 'fragments': fragments})

        return res

    def _interpret_outcome_split_in_sentences(self, to_predict):
        res = []
        for s_index in range(len(to_predict)):
            d = {}
            split_sentence_predictions = []
            sublist = merge_short_sentences(split_in_sentences(to_predict[s_index]))
            suboutcomes = super().predict(sublist)
            split_sentence_predictions.append((sublist, suboutcomes))
            article = ''
            fragments = []
            joined_outcome = ""
            for i, sub in enumerate(suboutcomes):
                if type(sub) == list:
                    #print("SUB = ", sub)
                    sub = sub[0]
                    joined_outcome += '\n' + sub

                sub_article, sub_fragments, _ = decode(sub, self.labels)

                # BEGIN alternative
                # tokenized_list = insert_tags_list(sublist[i], sub)
                # tagged_sentence = insert_tags(sublist[i], tokenized_list)
                # sub_article, sub_fragments, errors = decode(tagged_sentence, all_labels)
                # print("SENTENCES=[",sublist[i],"]")
                # print("OUT SENTENCE=[",sub,"]")
                # print("TAGGED SENTENCE=[",tagged_sentence, "]")
                # END alternative

                for sf in sub_fragments:
                    fragments.append(sf + len(article))  # Use __add__ function of Fragment
                article += sub_article

            for i in range(len(fragments)):
                fragments[i] = calibrate(fragments[i], article, to_predict[s_index], distance=10)

            # FIXME: now filtering out bad fragments, instead find out why they slip in
            fragments = [f for f in fragments if f.start < f.end]

            res.append({'id': '<UNKNOWN>', 'article': to_predict[s_index], 'fragments': fragments})

        return res

