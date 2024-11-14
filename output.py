from paddleocr import PaddleOCR, draw_ocr
import os
from natsort import natsorted
import json

ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=True,
                ocr_version='PP-OCRv4'
                )
result_dir = 'ocr\\python_test\\datasets\\cali_set_det'

# 强标注
img_dir = 'datasets\\cali_set_det'
threshold = 0.6
sort_img_dir = natsorted(os.listdir(img_dir))
train_data = {}

for filename in sort_img_dir:
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(img_dir, filename)
        try:
            result = ocr.ocr(img_path, cls=True)
            if result is None or len(result) == 0:
                print(f"文件 {filename} 的 OCR 结果为空，跳过该文件。")
                continue

            entries = []
            for line in result[0]:
                box = line[0]
                text = line[1][0]
                score = line[1][1]
                illegibility = score < threshold
                if illegibility:
                    continue
                entries.append({
                    "illegibility": illegibility,
                    "points": box,
                    "score": score,
                    "transcription": text
                })

            train_data[f"{os.path.splitext(filename)[0]}"] = entries

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

# 保存为 JSON 文件
with open('full_labels.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)