from collections import Counter, namedtuple
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import namedtuple, Counter
from xml.etree import ElementTree
import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]

    
def extract_sentences(path):
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    
    def catch_sen(obj):
        if obj is None:
            return []
        else:
            return obj.split(' ')
            
        
    def catch_num(obj):
        if obj is None:
            return []
            
        else:
            obj_update = obj.split(' ')
            obj_update_sec = [tuple(map(int, j.split(' '))) for j in [i.replace('-', ' ') for i in obj_update]]
            return obj_update_sec
        
    all_sentences = []
    all_targets = []
    
    xmlstring = open(path, 'r', encoding="utf8").read().replace("&", "&amp;")
    root = ElementTree.fromstring(xmlstring)

    for obj in root:
        
        # Составляю all_sentences
        eng_sen = catch_sen(obj[0].text)
        czech_sen = catch_sen(obj[1].text)
        pair_sen = SentencePair(eng_sen, czech_sen)
        all_sentences.append(pair_sen)
        
        # Составляю all_targets
        sure = catch_num(obj[2].text)
        possible = catch_num(obj[3].text)
        
        pair_numbers = LabeledAlignment(sure, possible)
        all_targets.append(pair_numbers)
        
    return all_sentences, all_targets



#def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    
    source_all_words = []
    target_all_words = []
    
    for pair in sentence_pairs:
        source_all_words.extend(pair.source)
        target_all_words.extend(pair.target)
    
    source_counter = Counter(source_all_words)
    target_counter = Counter(target_all_words)
    
    if freq_cutoff is None:
        source_dict = {j: i for i, j in enumerate(source_counter)}
        target_dict = {j: i for i, j in enumerate(target_counter)}
    
    else:   
        source_dict = {j[0]: i for i, j in enumerate(source_counter.most_common(freq_cutoff))}
        target_dict = {j[0]: i for i, j in enumerate(target_counter.most_common(freq_cutoff))}

    return source_dict, target_dict


def tokenize_sents(all_sentences, t_idx_src, t_idx_tgt):
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    output = []
    for pair in range(len(all_sentences)):
        eng_token = []
        crez_token = []
        for eng_word in all_sentences[pair].source:
            if eng_word in t_idx_src:
                eng_token.append(t_idx_src[eng_word])
            else:
                eng_token = []
                break
           
        for crez_word in all_sentences[pair].target:
            if crez_word in t_idx_tgt:
                crez_token.append(t_idx_tgt[crez_word])
            else:
                crez_token = []
                break
        
        #sentence_pairs = namedtuple('TokenizedSentencePair', 'source_tokens target_tokens')
        
        if len(eng_token) != 0 and len(crez_token) != 0:
            pair_class = TokenizedSentencePair(np.array(eng_token, dtype='int32'), np.array(crez_token, dtype='int32'))
            output.append(pair_class)

    return output

