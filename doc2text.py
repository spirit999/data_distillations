# 读取所有文件夹下的doc、docx、pdf文件，剔除掉所有的附件内容，保存为jsonl文件。
import os
import json
from pathlib import Path
import PyPDF2
from docx import Document
import subprocess
import tempfile
import re
import win32com.client

def remove_attachment_content(text):
    """
    通过识别附件标记来移除附件内容
    """
    if not text:
        return text

    attachment_patterns = [
        r'^附件[一二三四五六七八九十\d]+[\s:：]*[^\n]*\n',
        r'^附表[一二三四五六七八九十\d]+[\s:：]*[^\n]*\n',
        r'^附件[123456789\d]+[\s:：]*[^\n]*\n',
        r'^附件[一二三四五六七八九十]+[:：]',
        r'^附件[123456789]+[:：]',
        r'\n[【\[]?附件[】\]]?[\s\-]*\n',
        r'\n[【\[]?相关附件[】\]]?[\s\-]*\n',
        r'\n[【\[]?附件列表[】\]]?[\s\-]*\n',
        r'\n附件\s*[^\n]{0,50}\n',
        r'附件\s*[：:]\s*\d+[、,].*',
        r'附件\s*[：:].*?\d+[、,].*',
        r'附件\s*[：:].*?(?=\n\n|\Z)',
    ]

    attachment_positions = []
    for pattern in attachment_patterns:
        matches = list(re.finditer(pattern, text))
        if matches:
            # matches.sort()
            first_match = matches[0]
            attachment_positions.append(first_match.start())
    
    if not attachment_positions:
        return text

    candidate_pos = min(attachment_positions)
    text_after_candidate = text[candidate_pos:]
    lines_after = text_after_candidate.split('\n')
    
    if len(lines_after) > 20:  
        for i, pos in enumerate(sorted(attachment_positions)):
            lines_after_pos = text[pos:].split('\n')
            if len(lines_after_pos) <= 15:
                candidate_pos = pos
                break
    
    min_safe_length = int(len(text) * 0.2)
    actual_cut_pos = max(candidate_pos, min_safe_length)
    
    result = text[:actual_cut_pos].strip()
    if actual_cut_pos < len(text):
        original_length = len(text)
        result_length = len(result)
        removed_percentage = (original_length - result_length) / original_length * 100
        print(f"附件清理: 原文{original_length}字符 → 清理后{result_length}字符 (移除{removed_percentage:.1f}%)")
    return result

def read_pdf(file_path):
    """读取PDF文件内容，并移除附件部分"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ''
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            text = remove_attachment_content(text)
            return text.strip()
    except Exception as e:
        print(f"读取PDF文件 {file_path} 时出错: {e}")
        return ""

def read_docx(file_path):
    """读取DOCX文件内容，并移除附件部分"""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        text = '\n'.join(full_text)
        text = remove_attachment_content(text)
        return text
    except Exception as e:
        print(f"读取DOCX文件 {file_path} 时出错: {e}")
        return ""

def read_doc_linux(file_path):
    """
    读取DOC文件内容，并移除附件部分
    """
    try:
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                cmd = ['libreoffice', '--headless', '--convert-to', 'txt:Text', 
                       '--outdir', tmp_dir, file_path]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    base_name = Path(file_path).stem
                    txt_file = Path(tmp_dir) / f"{base_name}.txt"
                    
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        content = remove_attachment_content(content)
                        return content.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"LibreOffice转换失败: {e}")
        
        try:
            result = subprocess.run(['antiword', file_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"antiword转换失败: {e}")
        
        return f"无法处理DOC文件: {file_path}。请确保已安装LibreOffice或antiword。"
        
    except Exception as e:
        print(f"处理DOC文件 {file_path} 时出错: {e}")
        return ""

def get_all_files(directory):
    """递归获取目录下所有支持的文件"""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.doc', '.docx', '.pdf')):
                full_path = os.path.join(root, file)
                file_list.append(full_path)
    return file_list

def process_files_to_jsonl(input_folder, output_file):
    """处理文件并保存为JSONL格式"""
    is_windows = os.name == 'nt'
    if is_windows:
        print("检测到Windows系统，尝试安装pywin32以支持DOC文件...")
    
    all_files = get_all_files(input_folder)
    print(f"找到 {len(all_files)} 个文件需要处理")
    
    results = []
    for i, file_path in enumerate(all_files):
        print(f"处理进度: {i+1}/{len(all_files)} - {file_path}")
        text_content = ""
        file_ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)
        if file_ext == '.pdf':
            text_content = read_pdf(file_path)
            filename = filename.replace('（公开）', '').replace(".pdf", '').replace("\n\n",'')
        elif file_ext == '.docx':
            text_content = read_docx(file_path)
            filename = filename.replace('（公开）', '').replace(".docx", '').replace("\n\n",'')
        elif file_ext == '.doc':
            if is_windows:
                try:
                    text_content = read_doc_windows(file_path)
                    filename = filename.replace('（公开）', '').replace(".doc", '').replace("\n\n",'')
                except ImportError:
                    text_content = "请在Windows上安装pywin32以支持DOC文件: pip install pywin32"
            else:
                text_content = read_doc_linux(file_path)
                filename = filename.replace('（公开）', '').replace(".doc", '').replace("\n\n",'')
        
        file_data = {
            "id": file_path,
            "filename": filename,
            "text": text_content.replace('\n\n','')
        }
        results.append(file_data)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"处理完成！结果保存至: {output_file}")


def read_doc_windows(file_path):
    """Windows系统下使用win32com读取DOC文件"""
    try:
        word = win32com.client.Dispatch('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(file_path)
        text = doc.Content.Text
        doc.Close()
        word.Quit()
        return text.strip()
    except Exception as e:
        print(f"读取DOC文件 {file_path} 时出错: {e}")
        return ""


if __name__ == "__main__":
    input_folder = ""  
    output_file = ""
    if not os.path.exists(input_folder):
        print(f"错误: 文件夹 '{input_folder}' 不存在")
    else:
        process_files_to_jsonl(input_folder, output_file)