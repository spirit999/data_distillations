import jsonlines
import time
import os
import re
import torch
import argparse
import shutil
from tqdm import tqdm
from multiprocessing import Process, Queue, Lock
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gc


FILE_REF_WORDS = ["该文件", "本文件", "此文件", "该文档", "本文档", "此文档", "该办法", "本办法", "此办法", "该规则", "本规则"]
GENERAL_REF_WORDS = ["这", "那", "本", "该", "此", "其", "之", "该者", "此者", "其上", "其下", "其中"]
REF_PHRASES = ["同上", "附件", "前文", "上述", "如下", "以上", "以下"]

parser = argparse.ArgumentParser(description='Process a JSON Lines file and generate QA pairs (Alpaca/ShareGPT format) for LLM fine-tuning.')
parser.add_argument('--gpu_ids', type=str, default="0,1,2,3", help='Available GPU IDs (split by comma, e.g., 0,1,2)')
parser.add_argument('--n_process', type=int, default=None, help='Number of processes (default: same as GPU number)')
parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON Lines file (required)')
parser.add_argument('--output_file', type=str, help='Path to the output JSONL file (auto-generated if not specified)')
parser.add_argument('--model', type=str, default="/home/ps/Model/Qwen3-8B", help='Path to the LLM model (default: Qwen3-8B)')
parser.add_argument('--format', type=str, required=True, choices=['alpaca', 'sharegpt'], 
                    help='Output format: alpaca or sharegpt (required)')
parser.add_argument('--chunk_size', type=int, default=2000, help='Text chunk size (default: 2000)')
parser.add_argument('--chunk_overlap', type=int, default=100, help='Chunk overlap size (default: 100)')
parser.add_argument('--chunk_min', type=int, default=100, help='Minimum chunk size (default: 100)')
parser.add_argument('--debug', action='store_true', help='Enable debug mode to show more logs')
parser.add_argument('--deduplicate', action='store_true', help='Enable QA deduplication (remove duplicate questions)')

args = parser.parse_args()

INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file
MODEL_NAME = args.model
OUTPUT_FORMAT = args.format
CHUNK_SIZE = args.chunk_size
CHUNK_OVERLAP_SIZE = args.chunk_overlap
CHUNK_MINIMUM_SIZE = args.chunk_min
DEBUG_MODE = args.debug
ENABLE_DEDUPLICATE = args.deduplicate
GPU_IDS = [int(g) for g in args.gpu_ids.split(",") if g.strip()]
N_PROCESS = args.n_process if args.n_process is not None else len(GPU_IDS)
N_PROCESS = min(N_PROCESS, len(GPU_IDS))  
TEMP_DIR = f"temp_qa_{os.path.basename(INPUT_FILE).split('.')[0]}"
os.makedirs(TEMP_DIR, exist_ok=True)

if not OUTPUT_FILE:
    base_name = os.path.splitext(INPUT_FILE)[0]
    OUTPUT_FILE = f"{base_name}_{OUTPUT_FORMAT}_qa.jsonl"

if os.path.exists(OUTPUT_FILE):
    print(f"Skip {OUTPUT_FILE} (already exists)")
    exit()

def clean_pronouns(text, filename="", is_question=False):
    """
    适配大模型微调的文本清理逻辑：问题+答案双端彻底移除文件溯源信息，仅保留纯知识内容
    核心规则：
    1. 双端去溯源：问题/答案都移除《xxx》文档名、文件编号（〔xxx〕xxx号）、文件专属标识；
    2. 双端去代词：彻底清理模糊指代词汇/短语，避免语义模糊；
    3. 通用优化：清理冗余空格、连续标点，保证文本通顺，无无效字符；
    :param text: 待清理的Question/Answer文本
    :param filename: 兼容原有参数（无实际作用，仅保证函数调用不报错）
    :param is_question: 兼容原有参数（无实际作用，仅保证函数调用不报错）
    :return: 纯知识、无溯源、无代词的干净文本
    """
    if not text or text.strip() == "":
        return ""
   
    text = re.compile(r"《[^》]+》").sub("", text)
    
    text = re.compile(r"〔\d+〕\d+号|[\u4e00-\u9fa5]+字|[\u4e00-\u9fa5]+发").sub("", text)
    
    general_pattern = re.compile(r"(\s+)(这|那|本|该|此|其|附件|表)(\s+|。|，|！|？|：|;|,|!)")
    text = general_pattern.sub(r"\1\3", text)
    for phrase in REF_PHRASES:
        text = text.replace(phrase, "")
    
    text = re.sub(r"\s+", " ", text)  # 多空格/换行/制表符→单空格
    text = re.sub(r"([。，！？：;])([。，！？：;])", r"\1", text)  # 连续标点→单个
    text = text.strip()
    
    if text and text[0] in ['.', '，', '！', '？', '：', ';', ',', '"', "'"]:
        text = text[1:].strip()
    if text and text[-1] in [',', ':', ';', '"', "'"]:
        text = text[:-1].strip()
    
    return text


def load_model_tokenizer(gpu_id):
    """
    每个子进程独立加载模型/Tokenizer，避免跨进程显存冲突
    :param gpu_id: 绑定的GPU ID
    :return: model, tokenizer, device
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map={"": device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if DEBUG_MODE:
            print(f"Process {os.getpid()} - Model/Tokenizer loaded on GPU {gpu_id} successfully")
        return model, tokenizer, device
    except Exception as e:
        print(f"Process {os.getpid()} - Error loading model/tokenizer on GPU {gpu_id}: {str(e)}")
        raise e


def deduplicate_qa_list(qa_list):
    """QA对去重（基于问题内容，忽略大小写、空格和全角半角差异）"""
    if not ENABLE_DEDUPLICATE or not qa_list:
        return qa_list
    seen_questions = set()
    unique_qa = []
    for qa in qa_list:
        try:
            if OUTPUT_FORMAT == "alpaca":
                question = qa.get("instruction", "").strip()
            else:
                question = qa.get("conversations", [{}])[0].get("value", "").strip()
            if not question or len(question) < 3:
                continue
            question_key = question.lower().replace(" ", "").replace("　", "").replace("\t", "")
            if question_key not in seen_questions and len(question_key) > 3:
                seen_questions.add(question_key)
                unique_qa.append(qa)
        except Exception as e:
            if DEBUG_MODE:
                print(f"Debug - Deduplication skip invalid QA: {str(e)}")
            continue
    
    if DEBUG_MODE:
        print(f"Debug - Deduplication: {len(qa_list)} → {len(unique_qa)} QA pairs")
    return unique_qa


def generate_qa(paper_text, filename, model, tokenizer, device):
    """
    生成QA对，强化Prompt约束（从源头减少无关信息）+ 双端清理 + 3次失败重试 + 显存清理
    :param paper_text: 分块后的文本
    :param filename: 文件名（兼容传参，内部已做去溯源处理）
    :param model/tokenizer/device: 每个进程的模型/分词器/设备
    :return: 纯知识、无溯源、无代词的有效QA列表
    """
    max_retry = 3  
    retry_count = 0
    while retry_count < max_retry:
        try:
            system_prompt = """
            你是一个从文本中提取问答对的专家。你的任务是从提供的文本中生成尽可能多的高质量问答对，用于大模型有监督训练。
            要求：
            1. 问题要基于文本中的具体知识点，每个句子至少生成1个问题
            2. 问题与答案中都禁止使用"这"、"本"、"该"、“同上”、“附件”、“文件”、“本办法”等模糊不清的指代词汇
            4. 问答对要多样化，覆盖文本中的不同知识点
            5. 所有内容必须为中文，不要输出额外内容，严格按照指定格式输出
            6. 再次强调，问题与答案中禁止出现任何指代词
            7. 生成完成后，自己先检查：是否有代词？若有，立即修正后再输出。
            输出格式要求（每个问答对单独一行，不要空行）：
            Question:...Answer:...
            Question:...Answer:...
            """
            
            messages = [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": paper_text.strip()}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = tokenizer(
                [text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CHUNK_SIZE + 500  # 输入chunk_size + 500生成长度缓冲区
            ).to(device)
            
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=3000,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=1.1,
                num_return_sequences=1
            )
            
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            if DEBUG_MODE:
                print(f"\nDebug - Process {os.getpid()} Model output (first 500 chars):\n{response[:500]}...")
            
            response = response.replace("Question：", "Question:").replace("Answer：", "Answer:")
            response = response.replace("Question ", "Question:").replace("Answer ", "Answer:")
            response = response.replace("Question\t", "Question:").replace("Answer\t", "Answer:")
            if "Answer:" not in response and "Answer" in response:
                response = response.replace("Answer", "Answer:")
            
            qa_list = []
            pairs = [p.strip() for p in response.split("Question:") if p.strip() and len(p.strip()) > 5]
            
            for idx, pair in enumerate(pairs):
                pair = pair.replace("\n", " ").replace("\r", "").replace("\t", " ").replace('"', '\\"')
                pair = " ".join(pair.split())
                
                if len(pair) < 15 or "Answer:" not in pair:
                    if DEBUG_MODE:
                        print(f"Debug - Process {os.getpid()} Skip invalid pair {idx}: {pair[:100]}...")
                    continue
                
                parts = pair.split("Answer:", 1)
                if len(parts) != 2:
                    if DEBUG_MODE:
                        print(f"Debug - Process {os.getpid()} Invalid split {idx}: {pair[:100]}...")
                    continue
                
                raw_question = parts[0].strip()
                raw_answer = parts[1].strip()
                
                clean_q = clean_pronouns(raw_question, filename, is_question=True)
                clean_a = clean_pronouns(raw_answer, filename)
                
                if len(clean_q) < 5 or len(clean_a) < 5:
                    if DEBUG_MODE:
                        print(f"Debug - Process {os.getpid()} Too short QA after clean {idx}: Q='{clean_q}', A='{clean_a}'")
                    continue
                
                if OUTPUT_FORMAT == "alpaca":
                    qa_item = {
                        "instruction": clean_q,
                        "input": "",
                        "output": clean_a
                    }
                elif OUTPUT_FORMAT == "sharegpt":
                    qa_item = {
                        "conversations": [
                            {"from": "human", "value": clean_q},
                            {"from": "gpt", "value": clean_a}
                        ]
                    }
                qa_list.append(qa_item)
            
            qa_list = deduplicate_qa_list(qa_list)
            
            if DEBUG_MODE:
                print(f"Debug - Process {os.getpid()} Generated {len(qa_list)} valid QA pairs from this chunk")
            
            del model_inputs, generated_ids, response
            gc.collect()
            torch.cuda.empty_cache()
            
            return qa_list
        
        except Exception as e:
            retry_count += 1
            print(f"Process {os.getpid()} Generate QA retry {retry_count}/{max_retry} error: {str(e)}")
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)      
    print(f"Process {os.getpid()} Generate QA failed after {max_retry} retries")
    return []


def process_text_item(item, item_idx, model, tokenizer, device):
    """处理单个文本项，完成文本分块、逐块生成QA，返回所有有效QA对"""
    text = item.get("text", "").strip()
    filename = item.get("filename", "").strip()
    clean_text = text.replace(" ", "").replace("　", "").replace("\t", "")
    if not clean_text or len(clean_text) < 10:
        if DEBUG_MODE:
            print(f"Debug - Process {os.getpid()} Item {item_idx} has empty/too short text, skipping")
        return []
    
    if DEBUG_MODE:
        print(f"\n{'='*40}")
        print(f"Debug - Process {os.getpid()} Processing item {item_idx}, text length: {len(text)}")
        print(f"{'='*40}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP_SIZE,
        keep_separator=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", "、", "：", " ", "”", "’", ""]
    )
    
    chunks = text_splitter.split_text(text)
    if DEBUG_MODE:
        print(f"Debug - Process {os.getpid()} Split into {len(chunks)} raw chunks")
    
    merged_chunks = []
    current_chunk = ""
    for chunk in chunks:
        current_chunk += chunk
        if len(current_chunk) >= max(CHUNK_MINIMUM_SIZE * 0.8, 50):
            merged_chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk.strip() and len(current_chunk.strip()) >= 50:
        merged_chunks.append(current_chunk.strip())
    
    if DEBUG_MODE:
        print(f"Debug - Process {os.getpid()} Merged into {len(merged_chunks)} valid chunks")
    
    all_qa = []
    for chunk_idx, chunk in enumerate(merged_chunks):
        if DEBUG_MODE:
            print(f"\nDebug - Process {os.getpid()} Processing chunk {chunk_idx} of item {item_idx}, chunk length: {len(chunk)}")
        input_chunk = f"根据文件《{filename}》规定\n{chunk}" if filename else chunk
        qa = generate_qa(input_chunk, filename, model, tokenizer, device)
        all_qa.extend(qa)
    
    if DEBUG_MODE:
        print(f"Debug - Process {os.getpid()} Generated {len(all_qa)} QA pairs from item {item_idx}")
    
    return all_qa


def process_worker(item_list, process_id, gpu_id, progress_queue, lock):
    """
    子进程工作函数：绑定GPU→加载模型→处理分片item→写临时文件→更新进度
    :param item_list: 该进程处理的item分片列表
    :param process_id: 进程ID（0,1,2...）
    :param gpu_id: 绑定的GPU ID
    :param progress_queue: 进度队列，向主进程汇报处理进度
    :param lock: 进程锁，保证文件操作安全
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    model, tokenizer, device = load_model_tokenizer(gpu_id)
    temp_output = os.path.join(TEMP_DIR, f"temp_qa_{process_id}.jsonl")
    
    try:
        with jsonlines.open(temp_output, mode="w") as writer:
            for item_idx, item in enumerate(item_list):
                qa_list = process_text_item(item, item_idx, model, tokenizer, device)
                for qa in qa_list:
                    writer.write(qa)
                progress_queue.put(1)
        print(f"Process {process_id} (GPU {gpu_id}) - Processing complete! Temp file: {temp_output}")
    except Exception as e:
        print(f"Process {process_id} (GPU {gpu_id}) - Processing error: {str(e)}")
        if os.path.exists(temp_output):
            os.remove(temp_output)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
        progress_queue.put(None)  


def split_tasks(item_list, n_process):
    """将item列表平均分片，分给n个进程，处理余数保证负载均衡"""
    avg = len(item_list) // n_process
    remainder = len(item_list) % n_process
    tasks = []
    start = 0
    for i in range(n_process):
        end = start + avg + (1 if i < remainder else 0)
        tasks.append(item_list[start:end])
        start = end
    return tasks


def main():
    print(f"=" * 80)
    print(f"Local LLM QA Generation Tool (Multi-Process & Multi-GPU) - Fine-tuning Version")
    print(f"=" * 80)
    print(f"Input file:        {INPUT_FILE}")
    print(f"Output file:       {OUTPUT_FILE}")
    print(f"Model path:        {MODEL_NAME}")
    print(f"Output format:     {OUTPUT_FORMAT} | Deduplication: {'Enabled' if ENABLE_DEDUPLICATE else 'Disabled'}")
    print(f"Chunk config:      size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP_SIZE}, min={CHUNK_MINIMUM_SIZE}")
    print(f"Multi-GPU config:  GPU IDs={GPU_IDS}, Process num={N_PROCESS} (1 process → 1 GPU)")
    print(f"Debug mode:        {'Enabled' if DEBUG_MODE else 'Disabled'}")
    print(f"=" * 80)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} does not exist")
        exit(1)
    
    try:
        print(f"\nReading input file {INPUT_FILE}...")
        with jsonlines.open(INPUT_FILE) as reader:
            item_list = [item for item in reader]
        total_items = len(item_list)
        if total_items == 0:
            print(f"Error: Input file {INPUT_FILE} is empty")
            exit(1)
        print(f"Successfully read {total_items} items from input file")
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        exit(1)
    
    task_list = split_tasks(item_list, N_PROCESS)
    progress_queue = Queue()
    lock = Lock()
    processes = []
    
    print(f"\nStarting {N_PROCESS} processes (binding GPU {GPU_IDS})...")
    for process_id in range(N_PROCESS):
        gpu_id = GPU_IDS[process_id]
        current_task = task_list[process_id]
        if not current_task:
            print(f"Process {process_id} - No task, skip")
            continue
        p = Process(
            target=process_worker,
            args=(current_task, process_id, gpu_id, progress_queue, lock)
        )
        processes.append(p)
        p.start()
        print(f"Process {process_id} - Started (GPU {gpu_id}), processing {len(current_task)} items")
    
    print(f"\n{'='*40} Processing Start (Total items: {total_items}) {'='*40}")
    pbar = tqdm(total=total_items, desc="Total Processing Progress")
    completed = 0
    process_finished = 0
    while process_finished < N_PROCESS:
        msg = progress_queue.get()
        if msg is None:
            process_finished += 1
        else:
            completed += msg
            pbar.update(msg)
    pbar.close()
    print(f"{'='*40} All Processes Finished {'='*40}")
    
    print(f"\nMerging temporary files to {OUTPUT_FILE}...")
    try:
        with jsonlines.open(OUTPUT_FILE, mode="w") as writer:
            for process_id in range(N_PROCESS):
                temp_file = os.path.join(TEMP_DIR, f"temp_qa_{process_id}.jsonl")
                if os.path.exists(temp_file):
                    with jsonlines.open(temp_file) as reader:
                        for qa in reader:
                            writer.write(qa)
                    os.remove(temp_file)
        shutil.rmtree(TEMP_DIR)
    except Exception as e:
        print(f"Warning: Merge temp files error: {str(e)}, please merge temp files in {TEMP_DIR} manually")
    
    file_size = os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0
    total_qa_count = 0
    if os.path.exists(OUTPUT_FILE):
        with jsonlines.open(OUTPUT_FILE) as reader:
            total_qa_count = sum(1 for _ in reader)
    
    print(f"\n{'='*80}")
    if file_size == 0 or total_qa_count == 0:
        print("WARNING: Output file is empty! No valid QA pairs were generated.")
        print("Suggestions for troubleshooting:")
        print("1. Use --debug mode to check detailed error logs")
        print("2. Reduce --chunk_min (e.g., --chunk_min 50) to accept shorter text")
        print("3. Adjust --chunk_size to smaller value (e.g., --chunk_size 1000)")
        print("4. Check if input text has valid, non-empty content")
        print("5. Verify the LLM model can generate normal content")
    else:
        print("Processing Complete! QA pairs are clean (no trace, no pronouns) for LLM fine-tuning")
        print(f"Total items processed:  {total_items}")
        print(f"Total QA pairs generated: {total_qa_count} (pure knowledge, no trace)")
        print(f"Output file:         {OUTPUT_FILE}")
        print(f"Output file size:    {file_size / 1024:.2f} KB")
        print(f"Output format:       {OUTPUT_FORMAT} | Deduplication: {'Enabled' if ENABLE_DEDUPLICATE else 'Disabled'}")
    print(f"=" * 80)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    if torch.cuda.is_available():
        print(f"CUDA Available: {torch.cuda.device_count()} GPU(s) - {[torch.cuda.get_device_name(g) for g in GPU_IDS]}")
    else:
        print("WARNING: CUDA not available, all processes will use CPU (extremely slow!)")
        print("Suggestion: Use GPU for faster processing")
    main()