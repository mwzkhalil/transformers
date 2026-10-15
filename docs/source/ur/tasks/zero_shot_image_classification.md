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

# زیرو شاٹ تصویر کی درجہ بندی

[[open-in-colab]]

زیرو شاٹ امیج کی درجہ بندی ایک ایسا کام ہے جس میں ماڈل کا استعمال کرتے ہوئے تصاویر کو مختلف زمروں میں درجہ بندی کرنا شامل ہے جو ان مخصوص زمروں سے لیبل شدہ مثالوں پر مشتمل ڈیٹا پر واضح طور پر تربیت نہیں دی گئی تھی۔

روایتی طور پر، تصویر کی درجہ بندی کے لیے لیبل والی تصویروں کے مخصوص سیٹ پر ماڈل تربیت کی ضرورت ہوتی ہے، اور یہ ماڈل تصویر کی مخصوص خصوصیات کو لیبل پر "نقشہ" بنانا سیکھتا ہے۔ جب کسی درجہ بندی کے کام کے لیے اس طرح کے ماڈل کو استعمال کرنے کی ضرورت ہو جو لیبلز کا ایک نیا سیٹ متعارف کرائے، فائن ٹیوننگ کو ماڈل کو "دوبارہ کیلیبریٹ" کرنے کی ضرورت ہوتی ہے۔

اس کے برعکس، زیرو شاٹ یا کھلی الفاظ کی تصویر کی درجہ بندی ماڈلز عام طور پر ملٹی موڈل ماڈلز ہیں جنہیں امیجز اور متعلقہ وضاحتوں کے بڑے ڈیٹا سیٹ پر تربیت دی گئی ہے۔ یہ ماڈلز سیدھ میں وژن کی زبان کی نمائندگی سیکھتے ہیں جو زیرو شاٹ امیج کی درجہ بندی سمیت بہت سے نیچے والے کاموں کے لیے استعمال کیے جا سکتے ہیں۔

تصویر کی درجہ بندی کے لیے یہ ایک زیادہ لچکدار طریقہ ہے جو ماڈلز کو اضافی تربیتی ڈیٹا کی ضرورت کے بغیر نئے اور غیر دیکھے جانے والے زمروں کو عام کرنے کی اجازت دیتا ہے اور صارفین کو ان کے ہدف کی اشیاء کی مفت شکل کے متن کی وضاحت کے ساتھ تصاویر سے استفسار کرنے کے قابل بناتا ہے۔

اس گائیڈ میں آپ یہ سیکھیں گے کہ:

* ایک زیرو شاٹ امیج کی درجہ بندی پائپ لائن بنائیں
* زیرو شاٹ امیج کی درجہ بندی انفرنس ہاتھ سے چلائیں۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install -q "transformers[torch]" pillow
```

## زیرو شاٹ امیج کی درجہ بندی پائپ لائن

انفرنس کو آزمانے کا آسان طریقہ ماڈل کے ساتھ زیرو شاٹ امیج کی درجہ بندی کے لیے متعلقہ [`pipeline`] استعمال کرنا ہے۔ Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads) پر [چیک پوائنٹ سے ایک پائپ لائن شروع کریں:

```python
>>> from transformers import pipeline

>>> checkpoint = "openai/clip-vit-large-patch14"
>>> detector = pipeline(model=checkpoint, task="zero-shot-image-classification")
```

اگلا، ایک تصویر منتخب کریں جسے آپ درجہ بندی کرنا چاہتے ہیں۔

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/g8oS8-82DxI/download?ixid=MnwxMjA3fDB8MXx0b3BpY3x8SnBnNktpZGwtSGt8fHx8fDJ8fDE2NzgxMDYwODc&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/owl.jpg" alt="Photo of an owl"/>
</div>

تصویر اور امیدوار آبجیکٹ کے لیبل کو پائپ لائن پر منتقل کریں۔ یہاں ہم تصویر کو براہ راست منتقل کرتے ہیں۔ دیگر مناسب اختیارات میں ایک تصویر یا تصویری یو آر ایل کا مقامی راستہ شامل ہے۔ امیدوار لیبل سادہ الفاظ ہو سکتے ہیں جیسے اس مثال میں، یا زیادہ وضاحتی۔

```py
>>> predictions = detector(image, candidate_labels=["fox", "bear", "seagull", "owl"])
>>> predictions
[{'score': 0.9996670484542847, 'label': 'owl'},
 {'score': 0.000199399160919711, 'label': 'seagull'},
 {'score': 7.392891711788252e-05, 'label': 'fox'},
 {'score': 5.96074532950297e-05, 'label': 'bear'}]
```

## ہاتھ سے زیرو شاٹ امیج کی درجہ بندی

اب جب کہ آپ نے زیرو شاٹ امیج کی درجہ بندی پائپ لائن کو استعمال کرنے کا طریقہ دیکھا ہے، آئیے ایک نظر ڈالیں کہ آپ کس طرح زیرو شاٹ امیج کی درجہ بندی کو دستی طور پر چلا سکتے ہیں۔

ماڈل اور منسلک پروسیسر کو Hugging Face Hub](https://huggingface.co/models?pipeline_tag=zero-shot-image-classification&sort=downloads) پر [چیک پوائنٹ سے لوڈ کر کے شروع کریں۔ یہاں ہم پہلے کی طرح ہی چیک پوائنٹ استعمال کریں گے:

```py
>>> from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

>>> model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint)
>>> processor = AutoProcessor.from_pretrained(checkpoint)
```

آئیے چیزوں کو تبدیل کرنے کے لیے ایک مختلف تصویر لیں۔

```py
>>> from PIL import Image
>>> import requests

>>> url = "https://unsplash.com/photos/xBRQfR2bqNI/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjc4Mzg4ODEx&force=true&w=640"
>>> image = Image.open(requests.get(url, stream=True).raw)

>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" alt="Photo of a car"/>
</div>

ماڈل کے لیے ان پٹ تیار کرنے کے لیے پروسیسر کا استعمال کریں۔ پروسیسر ایک امیج پروسیسر کو جوڑتا ہے جو تصویر کو ماڈل کے لیے سائز تبدیل کرکے اور نارملائز کرکے تیار کرتا ہے، اور ایک ٹوکنائزر جو ٹیکسٹ ان پٹ کا خیال رکھتا ہے۔

```py
>>> candidate_labels = ["tree", "car", "bike", "cat"]
# follows the pipeline prompt template to get same results
>>> candidate_labels = [f'This is a photo of {label}.' for label in candidate_labels]
>>> inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)
```

ان پٹس کو ماڈل کے ذریعے منتقل کریں، اور نتائج کو پوسٹ پروسیس کریں:

```py
>>> import torch

>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits_per_image[0]
>>> probs = logits.softmax(dim=-1).numpy()
>>> scores = probs.tolist()

>>> result = [
...     {"score": score, "label": candidate_label}
...     for score, candidate_label in sorted(zip(probs, candidate_labels), key=lambda x: -x[0])
... ]

>>> result
[{'score': 0.998572, 'label': 'car'},
 {'score': 0.0010570387, 'label': 'bike'},
 {'score': 0.0003393686, 'label': 'tree'},
 {'score': 3.1572064e-05, 'label': 'cat'}]
```
