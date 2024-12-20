import os
import struct
import json

from tevatron.retriever.arguments import DataArguments


def get_jsonl_data_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size
    return int((fsize - 8) / 8)

def get_jsonl_data_item(fname, idx) :
    try :
        idx_name = fname + '.idx'
        with open(idx_name, 'rb') as h :
            h.seek((idx) * 8)
            start = struct.unpack('<Q', h.read(8))[0] # little endian, u64
            h.seek((idx + 1) * 8)
            end = struct.unpack('<Q', h.read(8))[0] # little endian, u64
        # print(f'start : { start }, end : { end }')
         
        with open(fname, 'rb') as h :
            h.seek(start)
            bdata = h.read(end - start)
            sdata = bdata.decode('utf8')
            json_data = json.loads(sdata)
            # json.dumps(sdata)
        return json_data # torch.tensor(token_data, dtype=torch.long), attention_mask
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, error : {str(e)} ')
        raise e

def get_jsonl_data_item_at(fname, start, len) :
    try :
        end = start + len
        # print(f'start : { start }, end : { end }')
         
        with open(fname, 'rb') as h :
            h.seek(start)
            bdata = h.read(end - start)
            sdata = bdata.decode('utf8')
            json_data = json.loads(sdata)
            # json.dumps(sdata)
        return json_data # torch.tensor(token_data, dtype=torch.long), attention_mask
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e

PASSAGES_PER_FILE = 2000000
def get_passage_at(lang, idx) :
    file_num = idx // PASSAGES_PER_FILE
    position = idx % PASSAGES_PER_FILE
    file_name = 'MSMARCO/' + lang + '/' + f'msmarco_passage_{file_num:02}'
    return get_jsonl_data_item(file_name, position)

def get_tsv_data_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size
    return int((fsize - 8) / 8)

def get_tsv_data_item(fname, idx, len) :
    try :
        idx_name = fname + '.idx'
        with open(idx_name, 'rb') as h :
            h.seek((idx) * 8)
            start = struct.unpack('<Q', h.read(8))[0] # little endian, u64
            h.seek((idx + 1) * 8)
            end = struct.unpack('<Q', h.read(8))[0] # little endian, u64
        # print(f'start : { start }, end : { end }')
         
        with open(fname, 'rb') as h :
            h.seek(start)
            bdata = h.read(end - start)
            sdata = bdata.decode('utf8').strip()
        return sdata.split('\t') # torch.tensor(token_data, dtype=torch.long), attention_mask
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e

def unescape_csv(bdata) :
    bcopy = bytearray()
    dquote = False

    for i, b in enumerate(bdata) :
        if b == 34 : # '"' is chr(34)
            if i == 0 or i == len(bdata) - 1 :
                continue
            elif dquote :
                dquote = False
                continue
            else :
                dquote = True
        elif dquote :
            dquote = False
        bcopy.append(b)

    return bcopy.decode('utf-8')

def get_csv_data_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size
    return int((fsize - 8) / 8) - 1 # csv header occupied one position

def get_csv_data_item(fname, idx, len) : # it's qid, query
    try :
        idx_name = fname + '.idx'
        with open(idx_name, 'rb') as h :
            h.seek((idx + 1) * 8)
            start = struct.unpack('>Q', h.read(8))[0] # csv index is big endian, u64
            if idx == len - 1 :
                end = -1
            else :
                h.seek((idx + 2) * 8)
                end = struct.unpack('>Q', h.read(8))[0] # csv index is big endian, u64
        # print(f'start : { start }, end : { end }')
        with open(fname, 'rb') as h :
            h.seek(start)
            if end >= 0 :
                row_data = h.read(end - start)
            else :
                row_data = h.read(os.stat(fname).st_size - start)
        
        row_text = unescape_csv(row_data[:-1])
        idx = row_text.index(',')
        
        return { 'qid' : row_text[:idx], 'query' : row_text[idx+1:] } # torch.tensor(token_data, dtype=torch.long), attention_mask
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e


import pandas as pd

root_dir = "/home/ubuntu/data/yoonforh"


queries_train_en_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_train_queries.tsv'), sep='\t', header=None, names=['qid', 'query'])

queries_train_ko_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_train_queries_ko.csv'))
qrels_train_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_train_qrels.tsv'), sep='\t', header=None, names=['qid', 'what', 'pid', 'relevance'])

queries_dev_en_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev_queries.tsv'), sep='\t', header=None, names=['qid', 'query'])
queries_dev_ko_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev_queries_ko.csv'))
qrels_dev_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev_qrels.tsv'), sep='\t', header=None, names=['qid', 'what', 'pid', 'relevance'])

queries_dev2_en_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev2_queries.tsv'), sep='\t', header=None, names=['qid', 'query'])
queries_dev2_ko_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev2_queries_ko.csv'))
qrels_dev2_df = pd.read_csv(os.path.join(root_dir, 'MSMARCO/passv2_dev2_qrels.tsv'), sep='\t', header=None, names=['qid', 'what', 'pid', 'relevance'])


# del passage_index_en, passage_index_ko

def load_json(file) :
    with open(file, 'rb') as f :
        return json.load(f)

# pid need to prepend 'msmarco_passage_' (16 chars)
# docid need to prepend 'msmarco_doc_' (12 chars)
# json is composed of <pid> : [ position, length, line_number, docid ] pairs

print("loading json...")
import time
start = time.time()
passage_index_en = load_json(os.path.join(root_dir, 'MSMARCO/en_index.json'))
print(f"{time.time()-start} seconds took loading en_index.json")
passage_index_ko = load_json(os.path.join(root_dir, 'MSMARCO/ko_index.json'))
print(f"{time.time()-start} seconds took loading ko_index.json")


from torch.utils.data import Dataset
import os
import struct
import torch
import random

class MarcoDataset(Dataset):
    def __init__(self, tokenizer, data_args: DataArguments, trainer = None):
        self.data_args = data_args
        
        self.tokenizer = tokenizer
        self.queries_en_df = queries_train_en_df
        self.queries_ko_df = queries_train_ko_df
        self.qrels_df = qrels_train_df
        self.real_len = len(queries_train_en_df)
        self.passage_len = len(passage_index_en)

        self.trainer = trainer
    
    def __len__(self):
        return self.real_len * 4 # we will use en-en, en-ko, ko-en, ko-ko combinations and other 4 is for negative samples

    def __getitem__(self, idx):
        real_idx = idx // 4
        lang_type = idx % 4

        ###
        epoch = int(self.trainer.state.epoch)
        _hashed_seed = hash(idx + self.trainer.args.seed)
        ###

        queries_df = self.queries_en_df if lang_type <= 1 else self.queries_ko_df
        passage_lang_en = True if lang_type == 0 or lang_type == 2 else False

        query = queries_df.iloc[real_idx]
        qid = query['qid']
        query_text = query['query']

        ### positive [0]
        
        # Find relevant passage ids for the query
        relevant_pids = self.qrels_df[self.qrels_df.qid == qid]['pid'].tolist()
        negative_pids = self.qrels_df[self.qrels_df.qid != qid]['pid'].tolist()

        if self.data_args.positive_passage_no_shuffle:
            pos_pid = relevant_pids[0]
        else:
            pos_pid = relevant_pids[(_hashed_seed + epoch) % len(relevant_pids)]
    
        passages = []

        # selecting negatives
        negative_size = self.data_args.train_group_size - 1
        if len(negative_pids) < negative_size:
            neg_pids = random.choices(negative_pids, k=negative_size)
        elif self.data_args.train_group_size == 1:
            neg_pids = []
        elif self.data_args.negative_passage_no_shuffle:
            neg_pids = negative_pids[:negative_size]
        else:
            _offset = epoch * negative_size % len(negative_pids)
            neg_pids = [x for x in negative_pids]
            random.Random(_hashed_seed).shuffle(neg_pids)
            neg_pids = neg_pids * 2
            neg_pids = neg_pids[_offset: _offset + negative_size]

        # get positive item
        index = pos_pid.rfind('_')

        if passage_lang_en :
            p_filename = '/home/ubuntu/data/yoonforh/MSMARCO/en/' + pos_pid[:index]
            p_tuple = passage_index_en.get(pos_pid[16:])
        else :
            p_filename = '/home/ubuntu/data/yoonforh/MSMARCO/ko/' + pos_pid[:index]
            p_tuple = passage_index_ko.get(pos_pid[16:])
        
        p_position = p_tuple[0]
        p_length = p_tuple[1]
        # p_line = p_tuple[2]
        # p_docid = p_tuple[3]

        p_text = get_jsonl_data_item_at(p_filename, p_position, p_length)['passage']
        passages.append(p_text)

        # get negative items
        for pid in neg_pids:
            index = pid.rfind('_')
            if passage_lang_en :
                p_filename = '/home/ubuntu/data/yoonforh/MSMARCO/en/' + pid[:index]
                p_tuple = passage_index_en.get(pid[16:])
            else :
                p_filename = '/home/ubuntu/data/yoonforh/MSMARCO/ko/' + pid[:index]
                p_tuple = passage_index_ko.get(pid[16:])
                
                p_position = p_tuple[0]
                p_length = p_tuple[1]
                # p_line = p_tuple[2]
                # p_docid = p_tuple[3]

                p_text = get_jsonl_data_item_at(p_filename, p_position, p_length)['passage']
                # passage_text = p_text
                passages.append(p_text)

        return query_text, passages


