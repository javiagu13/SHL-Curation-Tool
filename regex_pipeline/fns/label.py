# python label.py smc 2>&1 | tee -a logs/label.smc.log
import shutil
import os
import argparse
from functools import partial
import sys
from .elapsed import Elapsed
import yaml
from pathlib import Path
import regex as rx
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
import platform
import re
from .preprocessing import transform_arOfar_to_labeled_string, transform_to_labeled_string, regular_expression_tagger, parse_sentence_with_bio, bio_to_tagged, checkLength, wrap_labels, replace_date_string, remove_duplicates_and_special, predToTAG
from .tag2jsonl import parse_text_no_BI, parse_text, load_text_file_as_list, tag_to_BIO, save_list_as_text_file
###########################################
def load_config(configs):
    yml = ''
    for config in configs:
        with open(config, encoding='utf-8') as f:
            yml += f.read() + '\n'
    return yaml.load(yml, Loader=yaml.FullLoader)

###########################################
def find_pattern(pts, k, name='exception'):
    x = [p[k][name] for p in pts if k in p and name in p[k]]
    return x[0] if len(x) > 0 else None

###########################################
def add_label(line, colname, pattern, pts, company, formaton, flog):
    def _rep(x):
        # print(x)
        k, v = [(k, v) for k, v in x.groupdict().items() if v != None and not k.startswith('__')][0]

        ex = find_pattern(pts, k, 'exception')
        if ex is not None and rx.search(ex, v) is not None:
            return v

        it, nm = k.split('__', 1)
        if flog is not None:
            flog.write(f'\n{line}: {it}.{nm}: {v}')

        format_str = f' format="{nm}"' if formaton else ''
        if company == 'SMC':
            return f'<{it}{format_str}>{v}</{it}>'
        return f'<deid item="{it}"{format_str}>{v}</deid>'


    replaced = pattern.sub(_rep, line)
    if replaced != line and flog is not None:
        flog.write('\n')

    if flog is not None:
        org = line.split('\n')
        new = replaced.split('\n')
        for ii, line in enumerate(org):
            if line != new[ii]:
                flog.write(f'\t[{ii+1}]: {new[ii]}')
    return replaced

###########################################
def make_alias(cfg):
    pattern = rx.compile(r'\$\{[_\w]+\}')

    def _rxp(x):
        k = x.group()[2:-1]
        if k not in cfg['alias']:
            raise KeyError(f'No alias: {k}')

        v = cfg['alias'][k]
        m = pattern.search(v)
        return v if m == None else pattern.sub(_rxp, v)

    for k, v in cfg['alias'].items():
        cfg['alias'][k] = pattern.sub(_rxp, v)
    return cfg

###########################################
def get_patterns(cfg):
    _rxp = lambda x: cfg['alias'][x.group()[2:-1]]

    pattern = rx.compile(r'\$\{[_\w]+\}')

    rep = []
    for item in cfg['items']:
        it, fms = list(item.items())[0]
        for fm in fms:
            k, v = list(fm.items())[0]
            tmp = {}
            for x in ['pattern', 'exception']:
                if x not in v:
                    continue
                tmp[x] = pattern.sub(_rxp, v[x])
            rep.append({f'{it}__{k}': tmp})
    return rep

###########################################
def make_pattern(rep):
    pts = []
    for item in rep:
        k, v = list(item.items())[0]
        pts.append(f'(?P<{k}>{v["pattern"]})')
    return '|'.join(pts)

###########################################
def test_patterns(rep):
    ret = True
    for item in rep:
        k, v = list(item.items())[0]
        for x in ['pattern', 'exception']:
            if x not in v:
                continue
            try:
                # print(f'{k}: {v[x]}')
                # pattern = rx.compile(f'(?P<{k}>{v[x]})', flags=rx.I|rx.MULTILINE)
                pattern = rx.compile(f'(?P<{k}>{v[x]})', flags=rx.I)
            except rx.error as e:
                print(f'[E] {k}\n\t{x}: {v[x]}\n\t{e}')
                ret = False
                return False
    return ret
    
###########################################
def labeling_tqdm(lines, colname, pattern, pts, company, formaton, flog):
    return [add_label(line, colname, pattern, pts, company, formaton, flog) for line in lines]


# ###########################################
# def labeling(args, df):
#     df[args[0]] = df[args[0]].to_frame().apply(add_label, axis=1, args=args)
#     return df

###########################################
def parallel_df(df, fn, args):
    cpu_cnt = mp.cpu_count()
    df_split = np.array_split(df, cpu_cnt)
    pool = mp.Pool(cpu_cnt)
    fnp = partial(fn, args)
    df = pd.concat(pool.map(fnp, df_split))
    pool.close()
    pool.join()
    return df

###########################################
## JAVI ADDED FOR PARSING INITIAL TEXT
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

###########################################
def main(text, configs, findlog):
    # Your existing code
    lines=text.split('\n')
    
    elapsed = Elapsed()
    # print('config:', config)
    
    cfg = load_config(configs)
    company = cfg['company'] if 'company' in cfg else None
    formaton = cfg['format'] if 'format' in cfg else False

    path = Path(cfg['data']['root'], cfg['data']['in']).absolute()

    cfg = make_alias(cfg)
    pts = get_patterns(cfg)
    
    if not test_patterns(pts):
        sys.exit(9)

    pattern_str = make_pattern(pts)
    # print(pattern_str)
    
    # pattern = rx.compile(pattern_str, flags=rx.I|rx.MULTILINE)
    pattern = rx.compile(pattern_str, flags=rx.I)

    flog_root = Path(cfg['data']['root'], '_findlog')
    flog_root.mkdir(parents=True, exist_ok=True)



    #with open(path, 'r', encoding='utf-8') as file:
    #    lines = file.read().split('\n')

    #for c in df.columns:
    #    print('column:', c)
    c="column"
    f = None if not findlog else open(Path(flog_root, c+'.log'), 'w', encoding='utf-8')
    if platform.system() == 'Windows':
        print("lines")
        print(lines)
        lines = labeling_tqdm(lines, c, pattern, pts, company, formaton, f)
    else:
        # Modify this part based on your parallel processing requirements
        lines = parallel_processing_function(lines, labeling_tqdm, args=(c, pattern, pts, company, formaton, f))

    if f is not None:
        f.close()
        print('\tlog:', f.name)
    print(lines)

    ##Javi Preparing text for tokenization
    input_text=""
    for line in lines:
        input_text+=line+"\n"
    print("input_text")
    print(input_text)
    lines=read_input_text(input_text) #tokenization main algorithm

    ##PRED GENERATOR
    tags_to_check=["<DATE>", "</DATE>","<DATETIME>", "<OTHER_HOSPITAL>","</OTHER_HOSPITAL>", "<DATETIME>","</DATETIME>","<PERIOD>","</PERIOD>","<TIME>","</TIME>"]
    preds=[]
    for line in lines:
        pred_line=[]
        for word in line:
            if any(tag in word for tag in tags_to_check):
                print(word)
                # TAG detector
                pattern = r'<([^>]+)>|</([^>]+)>'

                # Use re.search to find the match
                match = re.search(pattern, word)
                if match:
                    # Check if the match corresponds to an opening tag or a closing tag
                    if match.group(1):
                        tag_name = match.group(1)

                if tag_name[0]=="/":
                    tag_name=tag_name[1:]
                print(f"The tag name is: {tag_name}")
                pred_line.append(tag_name)
            else:
                pred_line.append("O")
        preds.append(pred_line)
    print("input_text_preds")
    print(preds)

    ##TODO, delete all labels in the original text. from lines to cleaned_lines
    # Iterate through each array
    for array in lines:
        # Iterate through each string in the array
        for i, my_string in enumerate(array):
            #print(f"Original String {i + 1}: {my_string}")
            
            # Remove tags from the string
            for tag in tags_to_check:
                my_string = my_string.replace(tag, "")

            #print(f"Modified String {i + 1}: {my_string}")
            #print()
            # Update the string in the array
            array[i] = my_string
    print("output_lines")
    print(lines)


    ##TODO, copy function from BERT using preds and lines to give the response
    thor_text = str(transform_arOfar_to_labeled_string(lines,preds))

    return thor_text
###########################################

def predictREGEX(text, args):
    # Provide the required arguments for predictREGEX
    #args = Namespace(configs=["config/smc.yml"], findlog=True)
    
    _str2bool = lambda x: x.lower() in ['true', 'yes', 'y']
    
    parser = argparse.ArgumentParser()
    #parser.add_argument('configs',default="config/smc.yml", nargs='+')
    #parser.add_argument('-log', '--findlog', default=True, type=_str2bool)
    #args = parser.parse_args()
    result_doccano=main(text, args.configs, args.findlog)
    return result_doccano