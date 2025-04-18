import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

# 配置
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../docs'))
TEMP_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../temp_md_output'))
MAGIC_PDF_CMD = 'magic-pdf'
LANG = 'ch'
MODE = 'ocr'


def find_pdf_files(root_dir):
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(dirpath, fn))
    return pdf_files


def run_magic_pdf(pdf_path, output_dir):
    cmd = [
        MAGIC_PDF_CMD,
        '-p', pdf_path,
        '-o', output_dir,
        '-l', LANG,
        '-m', MODE
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def markdown_to_pdf(md_path, img_dir, output_pdf_path):
    """
    用playwright渲染markdown为pdf，支持本地图片。
    """
    # 生成临时html
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    # 简单md转html（可用markdown库增强）
    import markdown
    html_body = markdown.markdown(md_content, extensions=['extra', 'tables'])
    # 表格样式CSS
    table_css = '''
    <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
    }
    th, td {
      padding: 4px 8px;
    }
    </style>
    '''
    # 修正图片路径为file://
    html_body = html_body.replace('src="', f'src="file://{img_dir}/')
    html = f'<!DOCTYPE html><html><head>{table_css}</head><body>{html_body}</body></html>'
    html_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    html_file.write(html.encode('utf-8'))
    html_file.close()

    # 用playwright生成pdf
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(f'file://{html_file.name}')
        page.pdf(path=output_pdf_path, format='A4')
        browser.close()
    os.unlink(html_file.name)


def process_pdf(pdf_path, output_dir):
    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # 1. magic-pdf输出
    run_magic_pdf(pdf_path, output_dir)
    md_path = os.path.join(output_dir, pdf_name, 'ocr', f'{pdf_name}.md')
    img_dir = os.path.join(output_dir, pdf_name, 'ocr', 'images')
    if not os.path.exists(md_path):
        print(f"[Warning] Markdown not found for {pdf_path}")
        return
    # 2. 移动md到原目录
    new_md_path = os.path.join(pdf_dir, f'{pdf_name}.md')
    shutil.move(md_path, new_md_path)
    # 3. 渲染md为pdf
    rendered_pdf = os.path.join(pdf_dir, f'{pdf_name}.pdf')
    markdown_to_pdf(new_md_path, img_dir, rendered_pdf)
    # 4. 删除md文件
    if os.path.exists(new_md_path):
        os.remove(new_md_path)
    # 5. 清理magic-pdf输出（仅保留新pdf）
    shutil.rmtree(os.path.join(output_dir, pdf_name), ignore_errors=True)


def main():
    pdf_files = find_pdf_files(DOCS_DIR)
    if not pdf_files:
        print("No PDF files found.")
        return
    if os.path.exists(TEMP_OUTPUT):
        shutil.rmtree(TEMP_OUTPUT)
    os.makedirs(TEMP_OUTPUT, exist_ok=True)

    for pdf_path in pdf_files:
        try:
            process_pdf(pdf_path, TEMP_OUTPUT)
        except Exception as e:
            print(f"[Error] Processing {pdf_path}: {e}")

    if os.path.exists(TEMP_OUTPUT):
        shutil.rmtree(TEMP_OUTPUT)
    print("All PDFs processed: md+新pdf已生成，原pdf已替换。")


if __name__ == '__main__':
    main()
