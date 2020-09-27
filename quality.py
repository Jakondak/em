from typing import List, Tuple

from preprocessing import LabeledAlignment


def compute_precision(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the precision for predicted alignments.
    Numerator : |predicted and possible|
    Denominator: |predicted|
    Note that for correct metric values `sure` needs to be a subset of `possible`, but it is not the case for input data.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and possible sets, summed over all sentences
        total_predicted: total number of predicted alignments over all sentences
    """
    
    numerator = []
    denumerator = []
    
    for idx, pair in enumerate(reference):
        set_p = set(pair.possible)
        set_s = set(pair.sure)
        
        if set_s.issubset(set_p) == False:
            numerator_elem = len( (set_p.union(set_s)).intersection(set(predicted[idx])) )
            numerator.append(numerator_elem)
        else:
            numerator_elem = len( set_p.intersection(set(predicted[idx])) )
            numerator.append(numerator_elem)
            
        denumerator_elem = len(predicted[idx])
        denumerator.append(denumerator_elem)

    return sum(numerator), sum(denumerator)


def compute_recall(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> Tuple[int, int]:
    """
    Computes the numerator and the denominator of the recall for predicted alignments.
    Numerator : |predicted and sure|
    Denominator: |sure|

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        intersection: number of alignments that are both in predicted and sure sets, summed over all sentences
        total_predicted: total number of sure alignments over all sentences
    """
    
    numerator = []
    denumerator = []

    for idx, pair in enumerate(reference):
        numerator_elem = len(set(pair.sure).intersection(predicted[idx]))
        numerator.append(numerator_elem)
        denumerator_elem = len(pair.sure)
        denumerator.append(denumerator_elem)

    return sum(numerator), sum(denumerator)


def compute_aer(reference: List[LabeledAlignment], predicted: List[List[Tuple[int, int]]]) -> float:
    """
    Computes the alignment error rate for predictions.
    AER=1-(|predicted and possible|+|predicted and sure|)/(|predicted|+|sure|)
    Please use compute_precision and compute_recall to reduce code duplication.

    Args:
        reference: list of alignments with fields `possible` and `sure`
        predicted: list of alignments, i.e. lists of tuples (source_pos, target_pos)

    Returns:
        aer: the alignment error rate
    """
    num_prec, den_prec = compute_precision(reference, predicted)
    num_com, den_com = compute_recall(reference, predicted)

    aer = 1 - (num_prec+num_com) / (den_prec+den_com)

    return aer 
