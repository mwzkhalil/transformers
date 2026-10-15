<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# زیرو شاٹ آبجیکٹ کا پتہ لگانا

[[open-in-colab]]

روایتی طور پر، ماڈلز [آبجیکٹ کا پتہ لگانے](object_detection) کے لیے استعمال کیا جاتا ہے تربیت کے لیے لیبل والے تصویری ڈیٹاسیٹس کی ضرورت ہوتی ہے، اور یہ تربیتی ڈیٹا سے کلاسوں کے سیٹ کا پتہ لگانے تک محدود ہیں۔

زیرو شاٹ آبجیکٹ کا پتہ لگانا ایک کمپیوٹر وژن کا کام ہے جو کسی بھی پیشگی تربیت یا کلاسوں کی معلومات کے بغیر تصاویر میں اشیاء اور ان کی کلاسوں کا پتہ لگاتا ہے۔ زیرو شاٹ آبجیکٹ کا پتہ لگانا ماڈلز ان پٹ کے طور پر ایک تصویر کے ساتھ ساتھ امیدواروں کی کلاسوں کی فہرست حاصل کرتا ہے، اور باؤنڈنگ باکسز اور لیبلز کو آؤٹ پٹ کرتا ہے جہاں اشیاء کا پتہ چلا ہے۔

> [!نوٹ] > Hugging Face اس طرح کے بہت سے [اوپن ووکیبلری زیرو شاٹ آبجیکٹ ڈیٹیکٹر](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection) رکھتا ہے۔

اس گائیڈ میں، آپ اس طرح کے ماڈلز کو استعمال کرنے کا طریقہ سیکھیں گے:

- متن کے اشارے کی بنیاد پر اشیاء کا پتہ لگانے کے لیے
- بیچ آبجیکٹ کا پتہ لگانے کے لیے
- امیج گائیڈڈ آبجیکٹ کا پتہ لگانے کے لیے

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install -q transformers
```

## زیرو شاٹ آبجیکٹ کا پتہ لگانے والی پائپ لائن

انفرنس کو ماڈلز کے ساتھ آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-object-detection) پر [چیک پوائنٹ سے صفر شاٹ آبجیکٹ کا پتہ لگانے کے لیے ایک پائپ لائن کو فوری بنائیں:

```python
>>> from transformers import pipeline

>>> # Use any checkpoint from the hf.co/models?pipeline_tag=zero-shot-object-detection
>>> checkpoint = "iSEE-Laboratory/llmdet_large"
>>> detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
```

اس کے بعد، ایک تصویر منتخب کریں جس میں آپ اشیاء کا پتہ لگانا چاہتے ہیں۔ یہاں ہم خلاباز ایلین کولنز کی تصویر استعمال کریں گے جو کہ [NASA](https://www.nasa.gov/multimedia/imagegallery/index.html) عظیم امیجز ڈیٹاسیٹ کا حصہ ہے۔

```py
>>> from transformers.image_utils import load_image

>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png"
>>> image = load_image(url)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png" alt="Astronaut Eileen Collins"/>
</div>

پائپ لائن کو تلاش کرنے کے لیے تصویر اور امیدوار آبجیکٹ کے لیبل پاس کریں۔ یہاں ہم تصویر کو براہ راست منتقل کرتے ہیں۔ دیگر مناسب اختیارات میں ایک تصویر یا تصویری یو آر ایل کا مقامی راستہ شامل ہے۔ ہم ان تمام اشیاء کے لیے متن کی تفصیل بھی پاس کرتے ہیں جن کے لیے ہم تصویر سے استفسار کرنا چاہتے ہیں۔

```py
>>> predictions = detector(
...     image,
...     candidate_labels=["human face", "rocket", "nasa badge", "star-spangled banner"],
...     threshold=0.45,
... )
>>> predictions
[{'score': 0.8409242033958435,
  'label': 'human face',
  'box': {'xmin': 179, 'ymin': 74, 'xmax': 272, 'ymax': 179}},
 {'score': 0.7380027770996094,
  'label': 'rocket',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 466, 'ymax': 284}},
 {'score': 0.5850900411605835,
  'label': 'star-spangled banner',
  'box': {'xmin': 0, 'ymin': 0, 'xmax': 96, 'ymax': 511}},
 {'score': 0.5697067975997925,
  'label': 'human face',
  'box': {'xmin': 18, 'ymin': 15, 'xmax': 366, 'ymax': 511}},
 {'score': 0.47813931107521057,
  'label': 'star-spangled banner',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 459, 'ymax': 274}},
 {'score': 0.46597740054130554,
  'label': 'nasa badge',
  'box': {'xmin': 353, 'ymin': 0, 'xmax': 462, 'ymax': 279}},
 {'score': 0.4585932493209839,
  'label': 'nasa badge',
  'box': {'xmin': 132, 'ymin': 348, 'xmax': 208, 'ymax': 423}}]
```

آئیے پیشین گوئیوں کو دیکھتے ہیں:

```py
>>> from PIL import ImageDraw

>>> draw = ImageDraw.Draw(image)

>>> for prediction in predictions:
...     box = prediction["box"]
...     label = prediction["label"]
...     score = prediction["score"]

...     xmin, ymin, xmax, ymax = box.values()
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_2.png" alt="Visualized predictions on NASA image"/>
</div>

## متن سے اشارہ کیا گیا زیرو شاٹ آبجیکٹ کا ہاتھ سے پتہ لگانا

اب جب کہ آپ نے دیکھا ہے کہ زیرو شاٹ آبجیکٹ کا پتہ لگانے والی پائپ لائن کو کس طرح استعمال کرنا ہے، آئیے اسی نتیجہ کو دستی طور پر نقل کرتے ہیں۔

ماڈل اور منسلک پروسیسر کو Hugging Face Hub](https://hf.co/iSEE-Laboratory/llmdet_large) پر [چیک پوائنٹ سے لوڈ کر کے شروع کریں۔ یہاں ہم پہلے کی طرح ہی چیک پوائنٹ استعمال کریں گے:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

آئیے چیزوں کو تبدیل کرنے کے لیے ایک مختلف تصویر لیں۔

```py
>>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png"
>>> image = load_image(url)
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png" alt="Beach photo"/>
</div>

ماڈل کے لیے ان پٹ تیار کرنے کے لیے پروسیسر کا استعمال کریں۔

```py
>>> text_labels = ["hat", "book", "sunglasses", "camera"]
>>> inputs = processor(text=text_labels, images=image, return_tensors="pt").to(model.device)
```

ان پٹس کو ماڈل کے ذریعے منتقل کریں، عمل کے بعد، اور نتائج کا تصور کریں۔ چونکہ امیج پروسیسر نے تصاویر کو ماڈل میں فیڈ کرنے سے پہلے ان کا سائز تبدیل کیا ہے، اس لیے آپ کو `post_process_object_detection` طریقہ استعمال کرنے کی ضرورت ہے تاکہ یہ یقینی بنایا جا سکے کہ پیش گوئی شدہ باؤنڈنگ بکس میں اصل تصویر کے نسبت درست نقاط موجود ہیں:

```py
>>> import torch

>>> with torch.inference_mode():
...     outputs = model(**inputs)

>>> results = processor.post_process_grounded_object_detection(
...    outputs, threshold=0.50, target_sizes=[(image.height, image.width)], text_labels=text_labels,
... )[0]

>>> draw = ImageDraw.Draw(image)

>>> scores = results["scores"]
>>> text_labels = results["text_labels"]
>>> boxes = results["boxes"]

>>> for box, score, text_label in zip(boxes, scores, text_labels):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
...     draw.text((xmin, ymin), f"{text_label}: {round(score.item(),2)}", fill="white")

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## بیچ پروسیسنگ

آپ متعدد تصاویر میں مختلف (یا ایک جیسی) اشیاء کو تلاش کرنے کے لیے تصاویر کے متعدد سیٹ اور متن کے سوالات کو پاس کر سکتے ہیں۔ آئیے ایک خلاباز کی تصویر اور ساحل سمندر کی تصویر دونوں کو ایک ساتھ استعمال کریں۔ بیچ پروسیسنگ کے لیے، آپ کو ٹیکسٹ سوالات کو نیسٹڈ لسٹ کے طور پر پروسیسر اور امیجز کو PIL امیجز، PyTorch ٹینسر، یا NumPy arrays کی فہرست کے طور پر منتقل کرنا چاہیے۔

```py
>>> url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_1.png"
>>> url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_3.png"
>>> images = [load_image(url1), load_image(url2)]
>>> text_queries = [
...     ["human face", "rocket", "nasa badge", "star-spangled banner"],
...     ["hat", "book", "sunglasses", "camera", "can"],
... ]
>>> inputs = processor(text=text_queries, images=images, return_tensors="pt", padding=True)
```

پہلے پوسٹ پروسیسنگ کے لیے آپ نے سنگل امیج کے سائز کو ٹینسر کے طور پر پاس کیا تھا، لیکن آپ ایک ٹیوپل بھی پاس کر سکتے ہیں، یا، کئی امیجز کی صورت میں، ٹوپلز کی فہرست۔ آئیے دو مثالوں کے لیے پیشین گوئیاں بنائیں، اور دوسری مثال (`image_idx = 1`) کا تصور کریں۔

```py
>>> with torch.no_grad():
>>>     outputs = model(**inputs)

>>> target_sizes = [(image.height, image.width) for image in images]
>>> results = processor.post_process_grounded_object_detection(
...     outputs, threshold=0.3, target_sizes=target_sizes, text_labels=text_labels,
... )
```

آئیے نتائج کو دیکھتے ہیں:

```py
>>> image_idx = 1
>>> draw = ImageDraw.Draw(images[image_idx])

>>> scores = results[image_idx]["scores"].tolist()
>>> text_labels = results[image_idx]["text_labels"]
>>> boxes = results[image_idx]["boxes"].tolist()

>>> for box, score, text_label in zip(boxes, scores, text_labels):
>>>     xmin, ymin, xmax, ymax = box
>>>     draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
>>>     draw.text((xmin, ymin), f"{text_label}: {round(score,2)}", fill="white")

>>> images[image_idx]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_4.png" alt="Beach photo with detected objects"/>
</div>

## امیج گائیڈڈ آبجیکٹ کا پتہ لگانا

متن کے سوالات کے ساتھ زیرو شاٹ آبجیکٹ کا پتہ لگانے کے علاوہ، ماڈلز جیسے [OWL-ViT](https://huggingface.co/collections/ariG23498/owlvit-689b0d0872a7634a6ea17ae7) اور [OWLv2](https://huggingface.co/collections/ariG23498/owlv2-689b0d27bd7d96ba3c7f7530) امیج گائیڈڈ آبجیکٹ کا پتہ لگانے کی پیشکش کرتا ہے۔ اس کا مطلب ہے کہ آپ ٹارگٹ امیج میں ملتی جلتی اشیاء کو تلاش کرنے کے لیے تصویری استفسار استعمال کر سکتے ہیں۔

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

>>> checkpoint = "google/owlv2-base-patch16-ensemble"
>>> model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint, device_map="auto")
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

متن کے سوالات کے برعکس، صرف ایک مثال کی تصویر کی اجازت ہے۔

آئیے ایک صوفے پر دو بلیوں کے ساتھ ایک ٹارگٹ امیج کے طور پر اور ایک سوال کے طور پر ایک بلی کی تصویر لیں:

```py
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> image_target = Image.open(requests.get(url, stream=True).raw)

>>> query_url = "http://images.cocodataset.org/val2017/000000524280.jpg"
>>> query_image = Image.open(requests.get(query_url, stream=True).raw)
```

آئیے تصاویر پر ایک سرسری نظر ڈالتے ہیں:

```py
>>> import matplotlib.pyplot as plt

>>> fig, ax = plt.subplots(1, 2)
>>> ax[0].imshow(image_target)
>>> ax[1].imshow(query_image)
>>> fig.show()
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_5.png" alt="Cats"/>
</div>

پری پروسیسنگ مرحلے میں، ٹیکسٹ سوالات کے بجائے، اب آپ کو `query_images` استعمال کرنے کی ضرورت ہے:

```py
>>> inputs = processor(images=image_target, query_images=query_image, return_tensors="pt")
```

پیشین گوئیوں کے لیے، ان پٹس کو ماڈل پر منتقل کرنے کے بجائے، انہیں [`~OwlViTForObjectDetection.image_guided_detection`] پر منتقل کریں۔ پہلے کی طرح پیشین گوئیاں بنائیں سوائے اس کے کہ اب کوئی لیبل نہیں ہے۔

```py
>>> with torch.no_grad():
...     outputs = model.image_guided_detection(**inputs)
...     target_sizes = torch.tensor([image_target.size[::-1]])
...     results = processor.post_process_image_guided_detection(outputs=outputs, target_sizes=target_sizes)[0]

>>> draw = ImageDraw.Draw(image_target)

>>> scores = results["scores"].tolist()
>>> boxes = results["boxes"].tolist()

>>> for box, score in zip(boxes, scores):
...     xmin, ymin, xmax, ymax = box
...     draw.rectangle((xmin, ymin, xmax, ymax), outline="white", width=4)

>>> image_target
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/zero-sh-obj-detection_6.png" alt="Cats with bounding boxes"/>
</div>
