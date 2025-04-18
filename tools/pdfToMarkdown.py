import os
import shutil
import subprocess

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


def move_and_cleanup(pdf_path, output_dir):
    """
    查找magic-pdf输出的md文件，移动到pdf原位置并重命名，删除pdf和中间目录
    """
    pdf_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    # magic-pdf输出目录结构为 output_dir/pdf_name/ocr/pdf_name.md
    md_path = os.path.join(output_dir, pdf_name, 'ocr', f'{pdf_name}.md')
    if not os.path.exists(md_path):
        print(f"[Warning] Markdown not found for {pdf_path}")
        return
    new_md_path = os.path.join(pdf_dir, f'{pdf_name}.md')
    shutil.move(md_path, new_md_path)
    # 删除原pdf
    os.remove(pdf_path)
    # 删除magic-pdf生成的目录
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
            run_magic_pdf(pdf_path, TEMP_OUTPUT)
            move_and_cleanup(pdf_path, TEMP_OUTPUT)
        except Exception as e:
            print(f"[Error] Processing {pdf_path}: {e}")

    # 清理临时输出目录
    if os.path.exists(TEMP_OUTPUT):
        shutil.rmtree(TEMP_OUTPUT)
    print("All PDFs processed and replaced by markdown.")


if __name__ == '__main__':
    main()
