from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
from PIL import Image
from setuptools import setup
from io import open
import re
from paddleocr import PaddleOCR, draw_ocr

# Initialize OCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')

# Initialize deplot model
deplot_model = Pix2StructForConditionalGeneration.from_pretrained('google/deplot')
deplot_processor = Pix2StructProcessor.from_pretrained('google/deplot')

# Image path
image_path = "normal_chart.png"
image = Image.open(image_path)

# OCR model processes the image
ocr_output = ocr_model.ocr(image_path, cls=True)
for idx in range(len(ocr_output)):
    res = ocr_output[idx]
    for line in res:
        print(line)

# Extract text and numbers from OCR output
ocr_text = ' '.join([item[-1][0] for item in ocr_output[0]])
ocr_numbers = re.findall(r'\d+', ocr_text)

# Remove numbers from OCR output
other_text = re.sub(r'\d+', '', ocr_text)

# deplot model processes the image
inputs = deplot_processor(images=image, text="Describe this chart in detail:", return_tensors="pt")
predictions = deplot_model.generate(**inputs, max_new_tokens=512)
deplot_output = deplot_processor.decode(predictions[0], skip_special_tokens=True)

# Replace numbers in deplot output with numbers from OCR
text_number_pair = deplot_output
for number in ocr_numbers:
    text_number_pair = re.sub(r'\d+', number, text_number_pair, count=1)

print('Other Text:', other_text)
print('Text-Number Pair:', text_number_pair)