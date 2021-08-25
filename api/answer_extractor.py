from flask_restful import Resource, reqparse

import os
import re
import sys
import torch

import argparse
import torch

import os
# os.chdir(os.path.dirname(__file__))
sys.path.append(os.path.dirname(sys.path[0]))
print(os.getcwd())

from scripts import config, utils
from bertkpe import dataloader, tokenizer_class
# sys.path.append("scripts")
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'scripts'))
from scripts.model import KeyphraseSpanExtraction
from scripts.test import select_decoder
from scripts.utils import pred_arranger

from scripts.model import KeyphraseSpanExtraction
from scripts.test import bert2rank_decoder
import scripts.utils

def tokenize_fn(text):
    '''
    The tokenizer used in Meng et al. ACL 2017
    parse the feed-in text, filtering and tokenization
    keep [_<>,\(\)\.\'%], replace digits to 'DIGIT', split by [^a-zA-Z0-9_<>,\(\)\.\'%]
    :return: a list of tokens
    '''
    DIGIT = 'DIGIT'
    # remove line breakers
    text = re.sub(r'[\r\n\t]', ' ', text)
    # pad spaces to the left and right of special punctuations
    text = re.sub(r'[_<>,\(\)\.\'%]', ' \g<0> ', text)
    # tokenize by non-letters (new-added + # & *, but don't pad spaces, to make them as one whole word)
    tokens = filter(lambda w: len(w) > 0, re.split(r'[^a-zA-Z0-9_<>,#&\+\*\(\)\.\'%]', text))
    
    # replace the digit terms with DIGIT
    tokens = [w if not re.match('^\d+$', w) else DIGIT for w in tokens]

    return tokens

def preprocess(paragraph):

    tokens = tokenize_fn(paragraph)
 
    data = {}
    data['url'] = 0
    data['doc_words'] = tokens
    result = []
    result.append(data)
    return result


class ExtractAnswer(Resource):
    def post(self):
        parser = argparse.ArgumentParser('BertKPE', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        config.add_default_args(parser)
        args = parser.parse_args(['--no_cuda', '--run_mode', 'test', '--local_rank', '-1', '--model_class', 'bert2joint',
            '--pretrain_model_type', 'bert-base-multilingual-cased', '--dataset_class', 'squad', '--per_gpu_test_batch_size', '8',
            '--preprocess_folder', '', '--pretrain_model_path', '',
            '--eval_checkpoint', '"E:/TEMP/Multilingual-BERT-KPE/checkpoints/bert2joint.squad.bert.epoch_18'])

        config.init_args_config(args)
        saved_params = torch.load("E:/TEMP/Multilingual-BERT-KPE/checkpoints/bert2joint.squad.bert.epoch_18.checkpoint", map_location=lambda storage, loc:storage)
        
        args = saved_params['args']
        checkpoint_epoch = saved_params['epoch']
        state_dict = saved_params['state_dict']  
        model = KeyphraseSpanExtraction(args, state_dict)
        # model, checkpoint_epoch = KeyphraseSpanExtraction.load_checkpoint(args.eval_checkpoint, args)
        model.set_device()

        body_parser = reqparse.RequestParser()
        body_parser.add_argument('paragraph', required=True)
        body = body_parser.parse_args()

        # paragraph = "Saya adalah anak gembala yang suka nyanyi dan makan. Saya juga merupakan seorang yang suka tidur"
        # data = preprocess(paragraph)
        data = preprocess(body.paragraph)

        _, batchify_features_for_test = dataloader.get_class(args.model_class)
        tokenizer = tokenizer_class[args.pretrain_model_type].from_pretrained(args.cache_dir)
        print("mari dicek", data != None, data)
        dev_dataset = dataloader.build_dataset(**{'args':args, 'tokenizer':tokenizer, 'mode':'dev', 'example':data})
        args.test_batch_size = args.per_gpu_test_batch_size * max(1, args.n_gpu)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_data_loader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=args.test_batch_size,
            sampler=dev_sampler,
            num_workers=args.data_workers,
            collate_fn=batchify_features_for_test,
            pin_memory=args.cuda,
        )
        _, test_input_refactor = utils.select_input_refactor("bert2joint")

        decoder = select_decoder(args.model_class)
        pred = decoder(args, dev_data_loader, dev_dataset, model, test_input_refactor, pred_arranger, 'dev')

        keyphrases = pred[0]['KeyPhrases']

        return {'keyphrases': keyphrases}, 200