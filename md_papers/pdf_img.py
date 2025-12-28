import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_dir):
    # 1. 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建目录: {output_dir}")

    # 2. 打开 PDF 文件
    pdf_file = fitz.open(pdf_path)
    
    image_count = 0

    # 3. 遍历每一页
    for page_index in range(len(pdf_file)):
        page = pdf_file[page_index]
        image_list = page.get_images(full=True)

        # 如果当前页面没有图片，跳过
        if not image_list:
            continue

        print(f"正在处理第 {page_index + 1} 页，发现 {len(image_list)} 张图片...")

        # 4. 遍历页面中的所有图片对象
        for img_index, img in enumerate(image_list):
            xref = img[0]  # 图片的 XREF 编号
            base_image = pdf_file.extract_image(xref)
            
            image_bytes = base_image["image"]  # 图片二进制流
            image_ext = base_image["ext"]      # 图片扩展名 (png, jpeg 等)

            # 5. 构建保存路径
            image_filename = f"page{page_index+1}_img{img_index+1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            # 6. 写入文件
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            
            image_count += 1

    pdf_file.close()
    print("-" * 30)
    print(f"提取完成！共提取 {image_count} 张图片。")
    print(f"保存路径: {os.path.abspath(output_dir)}")

# --- 使用示例 ---
# 替换为你自己的 PDF 路径和想要保存的文件夹
pdf_input = "/root/workspace/homework/final_hw/SurveyX/md_papers/hallucination/Can We Trust AI Doctors A Survey of Medical Hallucination in Large.pdf" 
output_folder = "/root/workspace/homework/final_hw/SurveyX/md_papers/imgs"

extract_images_from_pdf(pdf_input, output_folder)