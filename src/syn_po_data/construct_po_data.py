import os
import re
import json
import random
import string
import argparse
from transformers import AutoTokenizer
from template import PROMPT_DICT
from collections import Counter

random.seed(42)

ans_prefixes = [
    "answer:",
    "the answer is:",
    "final answer is:",
]

def extract_answer(pred):
    pred = pred.lower()
    flag = False
    for prefix in ans_prefixes:
        idx = pred.rfind(prefix)
        if idx == -1:
            continue
        if len(pred) < idx + len(prefix) + 1:
            break
        ans = pred[idx + len(prefix):]
        flag = True
        return ans.strip(), flag
    return pred.strip(), flag

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def substring_exact_match_score(prediction, ground_truth):
    if isinstance(ground_truth, str):
        ground_truth = [ground_truth]
    for truth in ground_truth:
        norm_truth = normalize_answer(truth)
        norm_prediction = normalize_answer(prediction)
        if norm_truth in norm_prediction:
            return 1.0
    return 0.0

def qa_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def F1_scorer(prediction, ground_truths):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    score = 0.0
    for ground_truth in ground_truths:
        score = max(score, qa_f1_score(prediction, ground_truth))
    return score


def evaluate_output(output, ground_truth):
    pred_ans, flag = extract_answer(output)
    f1_score = 0
    subem_score = 0
    if flag:
        f1_score = F1_scorer(pred_ans, ground_truth)
        subem_score = substring_exact_match_score(pred_ans, ground_truth)
    return (subem_score + f1_score) / 2.0


def filter_metric(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    with open(args.input_dir, 'r', encoding='utf-8') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]
    print(len(data))

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    output_file = open(args.output_dir, 'w')

    for index, example in enumerate(data):
        ground_truth = example["answer"]
        all_predictions = []

        if "samples" in example:
            mab_preds = [example["samples"][rollout_idx]["pred"] for rollout_idx in range(len(example["samples"]))]
            all_predictions.extend(mab_preds)
        else:
            raise NotImplementedError("syn_po_data error!")

        scores = [evaluate_output(pred, ground_truth) for pred in all_predictions]

        max_score = max(scores)
        min_score = min(scores)

        if max_score == min_score:
            continue

        max_indices = [i for i, s in enumerate(scores) if s == max_score]
        min_indices = [i for i, s in enumerate(scores) if s == min_score]

        random.shuffle(max_indices)
        random.shuffle(min_indices)

        chosen = all_predictions[max_indices[0]]
        rejected = all_predictions[min_indices[0]]

        query = example['input']
        token_query = tokenizer([query])
        query_length = len(token_query.input_ids[0])

        chosen_len = len(tokenizer([chosen]).input_ids[0])
        rejected_len = len(tokenizer([rejected]).input_ids[0])
        max_response_len = max(chosen_len, rejected_len)

        context = example['context']
        token_context = tokenizer([context]).input_ids[0]
        context_len = len(token_context)

        budget = args.max_prompt_length - 32 - query_length - max_response_len
        if context_len > budget:
            split = budget // 2
            token_context = token_context[0:split] + token_context[-split:]
        new_context = tokenizer.decode(token_context, skip_special_tokens=True)
        input_data = PROMPT_DICT['cot_answer'].format(context=new_context, question=query)

        if index == 0:
            message = [
                {"role": "user", "content": input_data},
            ]
            prompt = tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True
            )
            print(prompt)

        dpo_example = {
            "id": example["id"],
            "input": example["input"],
            "answer": example["answer"],
            "prompt": input_data,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_reward": max_score,
            "rejected_reward": min_score,
            "chosen_index": max_indices[0],
            "rejected_index": min_indices[0],
        }
        output_file.write(json.dumps(dpo_example, ensure_ascii=False) + "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_prompt_length', type=int, default=8 * 1024)

    args = parser.parse_args()
    filter_metric(args)
