import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from .utils import init_logger, load_tokenizer, get_labels
from .javiPreprocessing import transform_arOfar_to_labeled_string, transform_to_labeled_string, regular_expression_tagger, parse_text, parse_sentence_with_bio, bio_to_tagged, checkLength, wrap_labels, replace_date_string, remove_duplicates_and_special, predToTAG
from .javiDeidentifyer import mask_labels, deidentify_dates_in_text, save_string_to_file, zip_folder
import time
import re 

logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_args.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        print("model directory:")
        print(args.model_dir)
        model = AutoModelForTokenClassification.from_pretrained("./bert_pipeline/model")#args.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)    #./bert_pipeline/model
        model.eval()        #./model
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config,input_text):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)
    print("READ INPUT FILES:")
    print(lines)
    return lines

def read_input_text(input_text):
    print("read_input_text_pre: ---------------------------------------------------")
    print(str(input_text))
    tokens=input_text.split(" ")
    line = []
    lines=[]
    current_token = ""

    for token in tokens:
        if token == "":
            current_token += ' '  # Concatenate spaces
        else:
            if current_token != "":
                line.append(current_token)  # Add accumulated spaces
                current_token = ""  # Reset for the next non-empty token
            if "\n" not in token:
                line.append(token)
            if "\n" in token:
                result = [match for match in re.split(r'(\n|\s+)', token) if match != '']
                for elem in result:
                    if elem=="\n":
                        lines.append(line)
                        line=[]
                    else:
                        line.append(elem)
            
    lines.append(line)
    print("read_input_text_post: ---------------------------------------------------")
    print(str(input_text))
    return lines

def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    print("lines")
    print(lines)
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset

def predict(pred_config, input_text):
    # load model and args
    args = get_args(pred_config)
    device = get_device(pred_config)
    model = load_model(pred_config, args, device)
    label_lst = get_labels(args)
    logger.info(args)
    nakedList=remove_duplicates_and_special(label_lst) #list just with unrepeated labels without O and without UNK, just labels such as: DATE, TIME...

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_text(input_text)
    #lines=linesInTAGtoBIOseq(lines,label_lst) #Javi parser from <TAG> to BIO, <-LOL THIS WAS NOT NECESSARY, SINCE WE DONT HAVE PREDICTIONS YET IM STUPID, this would be useful just for training directly from NAVER TAGS
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

    

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    preds = None

    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    preds = np.argmax(preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(label_lst)}
    preds_list = [[] for _ in range(preds.shape[0])]

    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                preds_list[i].append(slot_label_map[preds[i][j]])

    # Write to output file naver syle
    with open(pred_config.output_file_naver, "w", encoding="utf-8") as f:
        for words, preds in zip(lines, preds_list):
            line = ""
            for word, pred in zip(words, preds):
                if pred == 'O':
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            line=predToTAG(line,nakedList)
            f.write("{}\n".format(line.strip()))

    # Write to output file thor by lines
    # with open(pred_config.output_file_thor_lines, "w", encoding="utf-8") as f:
    #     countChar=0
    #     for words, preds in zip(lines, preds_list):
    #         f.write(str(transform_to_labeled_string(words,preds)))
    #         f.write("\n")
    # Write to output file thor all in one
    with open(pred_config.output_file_thor_all, "w", encoding="utf-8") as f:
        countChar=0
        totalwords=[]
        totalpreds=[]
        for words, preds in zip(lines, preds_list):
            totalwords.append(words)
            totalpreds.append(preds)
        print("totalwords")
        print(totalwords)
        print("totalpreds")
        print(totalpreds)
        thor_text = str(transform_arOfar_to_labeled_string(totalwords,totalpreds))
        #f.write("\n")

    logger.info("Prediction Done!")
    return thor_text


def predictBERT(input_text):
    print("HELLO CAN YOU READ THIS?")
    init_logger()
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--input_file", default="./bert_pipeline/sample_pred_in.txt", type=str, help="Input file for prediction")
    #./sample_pred_in.txt
    #./bert_pipeline/sample_pred_in.txt
    parser.add_argument("--output_file_naver", default="./bert_pipeline/JAVIS_Deidentification_Labeling/labeled_data/AI_labeled_naver_style.txt", type=str, help="Output file for prediction")
    parser.add_argument("--output_file_thor_lines", default="./bert_pipeline/JAVIS_Deidentification_Labeling/labeled_data/AI_labeled_thor_style_lines.jsonl", type=str, help="Output file for prediction")
    parser.add_argument("--output_file_thor_all", default="./bert_pipeline/JAVIS_Deidentification_Labeling/labeled_data/AI_labeled_thor_style_all.jsonl", type=str, help="Output file for prediction")

    parser.add_argument("--deidentified_file", default="./bert_pipeline/JAVIS_Deidentification_Labeling/deidentified_data/AI_deidentified_text.txt", type=str, help="deidentified file location")
    parser.add_argument("--masking_summary", default="./bert_pipeline/JAVIS_Deidentification_Labeling/deidentified_data/summary/1-mask_summary.txt", type=str, help="masking summary")
    parser.add_argument("--date_substitution_summary", default="./bert_pipeline/JAVIS_Deidentification_Labeling/deidentified_data/summary/2-date_substitution_summary.txt", type=str, help="date subsitution summary")
    parser.add_argument("--data_errors_summary", default="./bert_pipeline/JAVIS_Deidentification_Labeling/deidentified_data/summary/3-date_errors_summary.txt", type=str, help="date errors summary")

    parser.add_argument("--prediction_folder", default="./bert_pipeline/JAVIS_Deidentification_Labeling/", type=str, help="Output folder for prediction")


    

    parser.add_argument("--model_dir", default="./bert_pipeline/model", type=str, help="Path to save, load model")
    #./model
    #./bert_pipeline/model
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    text=predict(pred_config, input_text)
    return text
