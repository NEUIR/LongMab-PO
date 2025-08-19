
import os
import re
import json
import string
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Any
from template import PROMPT_DICT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader

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


class llmDataset(Dataset):

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def process_prompt(self, item):
        question = item['input']
        answer = item['answer']
        context = item['context']

        chunks = self.chunk_docs(content=context, chunk_size=args.chunk_size)

        template = PROMPT_DICT['gen_probe'].format(context=context, question=question, answer=answer)
        message = [
            {"role": "user", "content": template},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        item['input_prompt'] = [prompt]
        item['chunks'] = chunks

        return item

    def __getitem__(self, index):
        item = self.data[index]
        item = self.process_prompt(item)
        if index == 0:
            print(item)
        return item

    def __len__(self):
        return len(self.data)

    def chunk_docs(self, content, chunk_size, min_sentence=2, overlap=2):
        '''
            对长文本分块
            content: 长文本
            chunk_size: 分块最大长度（token）
            min_sentence: 分块最小长度（句子为单位）
            overlap: 分块之间重叠部分（句子为单位）
        '''
        stop_list = ['!', '。', '，', '！', '?', '？', ',', '.', ';']
        split_pattern = f"({'|'.join(map(re.escape, stop_list))})"
        sentences = re.split(split_pattern, content)

        if len(sentences) == 1:
            return sentences

        sentences = [sentences[i] + sentences[i + 1] for i in range(0, len(sentences) - 1, 2)]
        chunks = []
        temp_text = ''
        sentence_overlap_len = 0
        start_index = 0

        for i, sentence in enumerate(sentences):
            temp_text += sentence
            if self.get_prompt_length(temp_text) >= chunk_size - sentence_overlap_len or i == len(sentences) - 1:
                if i + 1 > overlap:
                    sentence_overlap_len = sum(
                        [self.get_prompt_length(sentences[j]) for j in range(i + 1 - overlap, i + 1)])
                if chunks:
                    if start_index > overlap:
                        start_index -= overlap
                chunk_text = ''.join(sentences[start_index:i + 1])
                if not chunks:
                    chunks.append(chunk_text)
                elif i == len(sentences) - 1 and (i - start_index + 1) < min_sentence:
                    chunks[-1] += chunk_text
                else:
                    chunks.append(chunk_text)
                temp_text = ''
                start_index = i + 1
        return chunks

    def get_prompt_length(self, text, **kwargs: Any):
        return len(self.tokenizer.encode(text, **kwargs))

    def Collactor(self, batch):
        id = [f['id'] for f in batch]
        input = [f['input'] for f in batch]
        context = [f['context'] for f in batch]
        chunks = [f['chunks'] for f in batch]
        input_prompt = [f['input_prompt'] for f in batch]
        answer = [f['answer'] for f in batch]
        num_tokens = [f['num_tokens'] for f in batch]

        return{
            'id': id,
            'input': input,
            'context': context,
            'chunks': chunks,
            'input_prompt': input_prompt,
            'answer': answer,
            'num_tokens': num_tokens,
        }

def generate_probe(args):

    with open(args.input_dir, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file][args.start:args.end]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = llmDataset(data, tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=args.n, collate_fn=dataset.Collactor)
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

    probe_data = []
    for batch in tqdm(dataloader):
        input_prompt = batch['input_prompt']
        flattened_input_prompt = [item for sublist in input_prompt for item in sublist]
        outputs: list = llm.generate(flattened_input_prompt, sampling_params)
        cleaned_outputs = [output.outputs[0].text for output in outputs]
        maxindex = len(batch['id'])
        for index in range(maxindex):
            id = batch['id'][index]
            input = batch['input'][index]
            context = batch['context'][index]
            chunks = batch['chunks'][index]
            answer = batch['answer'][index]
            num_tokens = batch['num_tokens'][index]
            pred = cleaned_outputs[index]
            match = re.search(r'the answer is (.*)', pred.lower())
            if match is None:
                continue
            else:
                pred_ans = match.group(1)
                if ((normalize_answer(answer) in normalize_answer(pred_ans))
                        or (normalize_answer(pred_ans) in normalize_answer(answer))):
                    output_item = {
                        "id": id,
                        "input": input,
                        "context": context,
                        "chunks": chunks,
                        "answer": answer,
                        "probe": pred,
                        "num_tokens": num_tokens,
                    }
                    probe_data.append(output_item)
    return probe_data


def compute_score(args, probe_data):

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    output_file = open(args.output_dir, 'w')

    model = SentenceTransformer(
        args.embedding_model_path,
        trust_remote_code=True,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.float16,
        }
    )

    with tqdm(total=len(probe_data), desc="Processing COT Data") as pbar:
        for item_idx, item in enumerate(probe_data):
            probe = item['probe']
            chunks = item['chunks']
            # 编码 probe 和 chunks
            probe_embedding = model.encode([probe]).tolist()
            chunk_embeddings = model.encode(chunks).tolist()

            # 转换为张量以便进行矩阵运算
            probe_tensor = torch.tensor(probe_embedding).cuda()
            chunks_tensor = torch.tensor(chunk_embeddings).cuda()

            # 归一化
            probe_tensor = F.normalize(probe_tensor, p=2, dim=1)
            chunks_tensor = F.normalize(chunks_tensor, p=2, dim=1)

            # 点积
            similarity_score = torch.matmul(probe_tensor, chunks_tensor.T)
            similarity_score = similarity_score.cpu().numpy().tolist()

            item['probe_similarity'] = similarity_score[0]
            json_str = json.dumps(item, ensure_ascii=False)
            output_file.write(json_str + "\n")
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--embedding_model_path', type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--chunk_size", type=int, default=1500)
    parser.add_argument("--max_model_len", type=int, default=30000)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=2000)
    args = parser.parse_args()

    print(args)
    # 生成探针
    probe_data = generate_probe(args)
    # 计算文本块重要性得分
    compute_score(args, probe_data)
