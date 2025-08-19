import os
import re
import json
import string
import argparse
import numpy as np
from collections import Counter
from template import PROMPT_DICT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
    norm_truth = normalize_answer(ground_truth)
    norm_prediction = normalize_answer(prediction)

    prediction_tokens = norm_prediction.split()
    ground_truth_tokens = norm_truth.split()

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
    if flag:
        f1_score = F1_scorer(pred_ans, ground_truth)
    subem_score = substring_exact_match_score(output, ground_truth)
    return (subem_score + f1_score) / 2.0


class UCBChunkSampler:
    def __init__(self, n_arms, init_scores):
        self.n = n_arms # chunk数量
        self.counts = np.zeros(n_arms)        # chunk已经被探索次数
        self.values = np.array(init_scores)   # chunk的期望价值，初始值根据初始化策略决定(random，similarity)
        self.total_rounds = 0 # 当前时间步rollout的轮数

    def sample_k(self, k):
        self.total_rounds += 1
        k = min(k, self.n)
        ucb_scores = np.zeros(self.n)
        for i in range(self.n):
            bonus = np.sqrt(2 * np.log(self.total_rounds) / (self.counts[i] + 1e-6))
            ucb_scores[i] = self.values[i] + bonus
        top_k = np.argpartition(ucb_scores, -k)[-k:]
        return top_k

    def update(self, selected_indices, reward):
        for index, i in enumerate(selected_indices):
            self.counts[i] += 1
            lr = 1.0 / (self.counts[i] + 1)
            self.values[i] = self.values[i] + lr * (reward - self.values[i])

def run_ucb_batch_sampling(llm, tokenizer, sampling_params, batch_data, rounds=30, k=1, method=None):
    """
        对样本数据按 batch_size 分批执行 UCB chunk 采样 + LLM 推理 + 奖励更新
    """
    for item in batch_data:
        if method == "probe_similarity":
            if "probe_similarity" in item:
                scores = item["probe_similarity"]
            else:
                raise NotImplementedError("score field error!")
        elif method == "query_similarity":
            if "query_similarity" in item:
                scores = item["query_similarity"]
            else:
                raise NotImplementedError("score field error!")
        elif method == "random":
            scores = np.random.uniform(0, 1, size=len(item["chunks"]))
        else:
            raise NotImplementedError("init score error!")
        item["sampler"] = UCBChunkSampler(len(item["chunks"]), scores)
        item["samples"] = []

    # 每条数据迭代 rounds 轮
    for round_id in range(rounds):
        print(f"⏳ Round {round_id + 1}/{rounds}")

        batch_prompts = []
        meta_info = []

        for item_id, item in enumerate(batch_data):
            sampler = item["sampler"]
            selected = sampler.sample_k(k)
            selected_chunks = [item["chunks"][i] for i in sorted(selected)]

            context = "\n".join(selected_chunks)
            template = PROMPT_DICT["cot_answer"].format(
                context=context,
                question=item["input"]
            )
            message = [{"role": "user", "content": template}]
            prompt = tokenizer.apply_chat_template(
                message, tokenize=False, add_generation_prompt=True
            )

            batch_prompts.append(prompt)
            meta_info.append({
                "selected": selected,
                "sampler": sampler,
                "ground_truth": item["answer"]
            })

        outputs = llm.generate(batch_prompts, sampling_params)

        for i, output in enumerate(outputs):
            pred = output.outputs[0].text
            reward = evaluate_output(pred, meta_info[i]['ground_truth'])

            sampler = meta_info[i]['sampler']
            sampler.update(meta_info[i]['selected'], reward)

            batch_data[i]["samples"].append({
                "round": round_id,
                "selected_indices": meta_info[i]["selected"].tolist(),
                "reward": reward,
                "score": sampler.values.tolist(),
                "pred": pred,
            })

    for item in batch_data:
        if isinstance(item, dict) and "sampler" in item:
            item.pop("sampler", None)

    return batch_data


def main(args):
    with open(args.input_dir, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]

    print(len(data))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    params_dict = {
        "n": 1,
        "best_of": 1,
        "presence_penalty": 1.0,
        "frequency_penalty": 0.0,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": -1,
        "use_beam_search": False,
        "length_penalty": 1,
        "early_stopping": False,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": False,
        "max_tokens": args.max_tokens,
        "logprobs": None,
        "prompt_logprobs": None,
        "skip_special_tokens": True,
    }
    sampling_params = SamplingParams(**params_dict)
    cuda_num = 1
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=cuda_num,
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len
    )

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(args.output_dir, 'w') as outfile:
        num_items = len(data)
        for start_idx in range(0, num_items, args.batch_size):
            end_idx = min(start_idx + args.batch_size, num_items)
            print(f"processing batch {end_idx}/{num_items}")
            batch_data = data[start_idx:end_idx]
            new_batch_data = run_ucb_batch_sampling(
                llm=llm,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                batch_data=batch_data,
                rounds=args.rounds,
                k=args.top_k,
                method=args.init_method
            )

            for item in new_batch_data:
                json_str = json.dumps(item, ensure_ascii=False)
                outfile.write(json_str + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--init_method", type=str, choices=["random", "probe_similarity", "query_similarity"], default="probe_similarity")
    parser.add_argument("--max_model_len", type=int, default=30000)
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    print(args)
    main(args)
