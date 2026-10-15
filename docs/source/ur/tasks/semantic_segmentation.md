<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# تصویر کی سیگمنٹیشن

[[open-in-colab]]

<Youtube id="dKE8SIt9C-w"/>

تصویر کی تقسیم ماڈلز تصویر میں دلچسپی کے مختلف شعبوں کے مطابق الگ الگ علاقے۔ یہ ماڈلز ہر پکسل کو ایک لیبل تفویض کر کے کام کرتے ہیں۔ سیگمنٹیشن کی کئی قسمیں ہیں: سیمنٹک سیگمنٹیشن، مثال سیگمنٹیشن، اور پینوپٹک سیگمنٹیشن۔

اس گائیڈ میں، ہم کریں گے:

1. مختلف اقسام کے سیگمنٹیشن](#types-of-segmentation) پر ایک نظر ڈالیں۔
2. [Semantic segmentation](#fine-tuning-a-model-for-segmentation) کے لیے اینڈ ٹو اینڈ فائن ٹیوننگ مثال رکھیں۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```py
# uncomment to install the necessary libraries
!pip install -q datasets transformers evaluate accelerate trackio
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

<a id='types-of-segmentation'></a>
## سیگمنٹیشن کی اقسام

سیمنٹک سیگمنٹیشن تصویر میں ہر ایک پکسل کو ایک لیبل یا کلاس تفویض کرتا ہے۔ آئیے ایک سیمینٹک سیگمنٹیشن ماڈل آؤٹ پٹ پر ایک نظر ڈالیں۔ یہ کسی تصویر میں آنے والی کسی چیز کی ہر مثال کو ایک ہی کلاس تفویض کرے گا، مثال کے طور پر، تمام بلیوں کو "کیٹ-1"، "کیٹ-2" کے بجائے "بلی" کا لیبل لگا دیا جائے گا۔ ہم Transformers' امیج سیگمنٹیشن پائپ لائن استعمال کر سکتے ہیں تاکہ فوری طور پر سیمنٹک سیگمنٹیشن ماڈل کا اندازہ لگایا جا سکے۔ آئیے مثال کی تصویر پر ایک نظر ڈالیں۔

```python
from transformers import pipeline
from PIL import Image
import requests

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg" alt="Segmentation Input"/>
</div>

ہم [nvidia/segformer-b1-finetuned-cityscapes-1024-1024](https://huggingface.co/nvidia/segformer-b1-finetuned-cityscapes-1024-1024) استعمال کریں گے۔

```python
semantic_segmentation = pipeline("image-segmentation", "nvidia/segformer-b1-finetuned-cityscapes-1024-1024")
results = semantic_segmentation(image)
results
```

سیگمنٹیشن پائپ لائن آؤٹ پٹ میں ہر پیش گوئی شدہ کلاس کے لیے ایک ماسک شامل ہوتا ہے۔

```bash
[{'score': None,
  'label': 'road',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'sidewalk',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'building',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'wall',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'pole',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'traffic sign',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'vegetation',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'terrain',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': None,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

کار کلاس کے ماسک پر ایک نظر ڈالتے ہوئے، ہم دیکھ سکتے ہیں کہ ہر کار کو ایک ہی ماسک کے ساتھ درجہ بندی کیا گیا ہے۔

```python
results[-1]["mask"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/semantic_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

مثال کی تقسیم میں، مقصد ہر پکسل کی درجہ بندی کرنا نہیں ہے، بلکہ دی گئی تصویر میں **کسی چیز کی ہر مثال** کے لیے ماسک کی پیشین گوئی کرنا ہے۔ یہ آبجیکٹ کا پتہ لگانے کی طرح کام کرتا ہے، جہاں ہر مثال کے لیے ایک باؤنڈنگ باکس ہوتا ہے، اس کے بجائے ایک سیگمنٹیشن ماسک ہوتا ہے۔ ہم اس کے لیے [facebook/mask2former-swin-large-cityscapes-instance](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-instance) استعمال کریں گے۔

```python
instance_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-instance")
results = instance_segmentation(image)
results
```

جیسا کہ آپ نیچے دیکھ سکتے ہیں، متعدد کاروں کی درجہ بندی کی گئی ہے، اور پکسلز کے علاوہ کسی دوسرے پکسلز کے لیے کوئی درجہ بندی نہیں ہے جو کار اور شخصی مثالوں سے تعلق رکھتے ہیں۔

```bash
[{'score': 0.999944,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999945,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999652,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.903529,
  'label': 'person',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

ذیل میں کار کے ماسک میں سے ایک کو چیک کر رہے ہیں۔

```python
results[2]["mask"]
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/instance_segmentation_output.png" alt="Semantic Segmentation Output"/>
</div>

Panoptic segmentation semantic segmentation اور instance segmentation کو یکجا کرتا ہے، جہاں ہر پکسل کو کلاس اور اس کلاس کی ایک مثال میں درجہ بندی کیا جاتا ہے، اور کلاس کی ہر مثال کے لیے متعدد ماسک ہوتے ہیں۔ ہم اس کے لیے [facebook/mask2former-swin-large-cityscapes-panoptic](https://huggingface.co/facebook/mask2former-swin-large-cityscapes-panoptic) استعمال کر سکتے ہیں۔

```python
panoptic_segmentation = pipeline("image-segmentation", "facebook/mask2former-swin-large-cityscapes-panoptic")
results = panoptic_segmentation(image)
results
```

جیسا کہ آپ نیچے دیکھ سکتے ہیں، ہمارے پاس مزید کلاسز ہیں۔ ہم بعد میں یہ دیکھیں گے کہ ہر پکسل کو کلاسوں میں سے ایک میں درجہ بندی کیا گیا ہے۔

```bash
[{'score': 0.999981,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999958,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.99997,
  'label': 'vegetation',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999575,
  'label': 'pole',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999958,
  'label': 'building',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999634,
  'label': 'road',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.996092,
  'label': 'sidewalk',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.999221,
  'label': 'car',
  'mask': <PIL.Image.Image image mode=L size=612x415>},
 {'score': 0.99987,
  'label': 'sky',
  'mask': <PIL.Image.Image image mode=L size=612x415>}]
```

آئیے ہر قسم کے سیگمنٹیشن کا ایک ساتھ موازنہ کرتے ہیں۔

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation-comparison.png" alt="Segmentation Maps Compared"/>
</div>

ہر قسم کے سیگمنٹیشن کو دیکھتے ہوئے، آئیے سیمنٹک سیگمنٹیشن کے لیے فائن ٹیوننگ a ماڈل پر گہرا غوطہ لگائیں۔

سیمنٹک سیگمنٹیشن کی عام حقیقی دنیا کی ایپلی کیشنز میں پیدل چلنے والوں اور ٹریفک کی اہم معلومات کی شناخت کے لیے خود چلانے والی کاروں کو تربیت دینا، طبی تصویروں میں خلیات اور اسامانیتاوں کی نشاندہی کرنا، اور سیٹلائٹ امیجری سے ماحولیاتی تبدیلیوں کی نگرانی کرنا شامل ہے۔

<a id='fine-tuning-a-model-for-segmentation'></a>
## سیگمنٹیشن کے لیے فائن ٹیوننگ a ماڈل

اب ہم کریں گے:

1. فائن ٹیون [SegFormer](https://huggingface.co/docs/transformers/main/en/model_doc/segformer#segformer) [SceneParse150](https://huggingface.co/datasets/merve/scene_parse_150) ڈیٹا سیٹ پر۔
2. انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/image-segmentation) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

### SceneParse150 ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے SceneParse150 ڈیٹاسیٹ کا ایک چھوٹا سب سیٹ لوڈ کرکے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/scene_parse_150", split="train[:50]")
```

ڈیٹا سیٹ کے `train` کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```py
>>> ds = ds.train_test_split(test_size=0.2)
>>> train_ds = ds["train"]
>>> test_ds = ds["test"]
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> train_ds[0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x683 at 0x7F9B0C201F90>,
 'annotation': <PIL.PngImagePlugin.PngImageFile image mode=L size=512x683 at 0x7F9B0C201DD0>,
 'scene_category': 368}

# view the image
>>> train_ds[0]["image"]
```

- `image`: منظر کی ایک PIL تصویر۔
- `annotation`: تقسیم کے نقشے کی ایک PIL تصویر، جو ماڈل کا ہدف بھی ہے۔
- `scene_category`: ایک زمرہ کی شناخت جو تصویر کے منظر کو بیان کرتی ہے جیسے "کچن" یا "آفس"۔ اس گائیڈ میں، آپ کو صرف `image` اور `annotation` کی ضرورت ہوگی، یہ دونوں PIL تصاویر ہیں۔

آپ ایک ایسی لغت بھی بنانا چاہیں گے جو لیبل کی شناخت کو لیبل کلاس میں نقشہ بنائے جو بعد میں ماڈل سیٹ اپ کرنے پر کارآمد ہوگی۔ حب سے نقشہ جات ڈاؤن لوڈ کریں اور `id2label` اور `label2id` لغات بنائیں:

```py
>>> import json
>>> from pathlib import Path
>>> from huggingface_hub import hf_hub_download

>>> repo_id = "huggingface/label-files"
>>> filename = "ade20k-id2label.json"
>>> id2label = json.loads(Path(hf_hub_download(repo_id, filename, repo_type="dataset")).read_text())
>>> id2label = {int(k): v for k, v in id2label.items()}
>>> label2id = {v: k for k, v in id2label.items()}
>>> num_labels = len(id2label)
```

### پری پروسیس

اگلا مرحلہ ماڈل کے لیے تصاویر اور تشریحات تیار کرنے کے لیے SegFormer امیج پروسیسر لوڈ کرنا ہے۔ کچھ ڈیٹا سیٹس، اس طرح، زیرو انڈیکس کو بیک گراؤنڈ کلاس کے طور پر استعمال کرتے ہیں۔ تاہم، پس منظر کی کلاس اصل میں 150 کلاسوں میں شامل نہیں ہے، لہذا آپ کو تمام لیبلز سے ایک کو گھٹانے کے لیے `do_reduce_labels=True` سیٹ کرنا ہوگا۔ صفر انڈیکس کو `255` سے بدل دیا گیا ہے لہذا اسے SegFormer کے نقصان کے فنکشن نے نظر انداز کر دیا ہے:

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "nvidia/mit-b0"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint, do_reduce_labels=True)
```

اوور فٹنگ کے خلاف ماڈل کو زیادہ مضبوط بنانے کے لیے تصویری ڈیٹاسیٹ میں کچھ ڈیٹا کو بڑھانا عام ہے۔ اس گائیڈ میں، آپ کسی تصویر کی رنگین خصوصیات کو تصادفی طور پر تبدیل کرنے کے لیے [torchvision](https://pytorch.org/vision/stable/generated/torchvision.transforms.ColorJitter.html) سے [`ColorJitter`](https://pytorch.org/vision/stable/index.html) فنکشن استعمال کریں گے، لیکن آپ اپنی پسند کی کوئی بھی تصویری لائبریری بھی استعمال کر سکتے ہیں۔

```py
>>> from torchvision.transforms import ColorJitter

>>> jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)
```

اب ماڈل کے لیے تصاویر اور تشریحات تیار کرنے کے لیے دو پری پروسیسنگ فنکشن بنائیں۔ یہ فنکشن تصاویر کو `pixel_values` اور تشریحات کو `labels` میں تبدیل کرتے ہیں۔ تربیتی سیٹ کے لیے، امیج پروسیسر کو تصاویر فراہم کرنے سے پہلے `jitter` لاگو کیا جاتا ہے۔ ٹیسٹ سیٹ کے لیے، امیج پروسیسر `images` کو تراشتا اور نارمل کرتا ہے، اور صرف `labels` کو تراشتا ہے کیونکہ ٹیسٹنگ کے دوران ڈیٹا میں کوئی اضافہ نہیں کیا جاتا ہے۔

```py
>>> def train_transforms(example_batch):
...     images = [jitter(x) for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs


>>> def val_transforms(example_batch):
...     images = [x for x in example_batch["image"]]
...     labels = [x for x in example_batch["annotation"]]
...     inputs = image_processor(images, labels)
...     return inputs
```

پورے ڈیٹا سیٹ پر `jitter` لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.set_transform`] فنکشن استعمال کریں۔ ٹرانسفارم کو فلائی پر لگایا جاتا ہے جو تیز ہے اور ڈسک کی کم جگہ استعمال کرتا ہے:

```py
>>> train_ds.set_transform(train_transforms)
>>> test_ds.set_transform(val_transforms)
```

### اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [میین انٹرسیکشن اوور یونین](https://huggingface.co/spaces/evaluate-metric/accuracy) (IoU) میٹرک کو لوڈ کریں (دیکھیں 🤗 اندازہ کریں [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے بارے میں مزید جاننے کے لیے):

```py
>>> import evaluate

>>> metric = evaluate.load("mean_iou")
```

پھر [`~evaluate.EvaluationModule.compute`] میٹرکس کے لیے ایک فنکشن بنائیں۔ آپ کی پیشین گوئیوں کو پہلے لاگٹس میں تبدیل کرنے کی ضرورت ہے، اور پھر آپ [`~evaluate.EvaluationModule.compute`] کو کال کرنے سے پہلے لیبل کے سائز سے مماثل ہونے کے لیے نئی شکل دینے کی ضرورت ہے:

```py
>>> import numpy as np
>>> import torch
>>> from torch import nn

>>> def compute_metrics(eval_pred):
...     with torch.no_grad():
...         logits, labels = eval_pred
...         logits_tensor = torch.from_numpy(logits)
...         logits_tensor = nn.functional.interpolate(
...             logits_tensor,
...             size=labels.shape[-2:],
...             mode="bilinear",
...             align_corners=False,
...         ).argmax(dim=1)

...         pred_labels = logits_tensor.detach().cpu().numpy()
...         metrics = metric.compute(
...             predictions=pred_labels,
...             references=labels,
...             num_labels=num_labels,
...             ignore_index=255,
...             reduce_labels=False,
...         )
...         for key, value in metrics.items():
...             if isinstance(value, np.ndarray):
...                 metrics[key] = value.tolist()
...         return metrics
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

### ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! SegFormer کو [`AutoModelForSemanticSegmentation`] کے ساتھ لوڈ کریں، اور ماڈل لیبل آئی ڈیز اور لیبل کلاسز کے درمیان میپنگ پاس کریں:

```py
>>> from transformers import AutoModelForSemanticSegmentation, TrainingArguments, Trainer

>>> model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, id2label=id2label, label2id=label2id)
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ یہ ضروری ہے کہ آپ غیر استعمال شدہ کالموں کو نہ ہٹائیں کیونکہ اس سے `image` کالم گر جائے گا۔ `image` کالم کے بغیر، آپ `pixel_values` نہیں بنا سکتے۔ اس رویے کو روکنے کے لیے `remove_unused_columns=False` سیٹ کریں! صرف دوسرا مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] IoU میٹرک کا جائزہ لے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="segformer-b0-scene-parse-150",
...     learning_rate=6e-5,
...     num_train_epochs=50,
...     per_device_train_batch_size=2,
...     per_device_eval_batch_size=2,
...     save_total_limit=3,
...     eval_strategy="steps",
...     save_strategy="steps",
...     save_steps=20,
...     eval_steps=20,
...     logging_steps=1,
...     report_to="trackio",
...     run_name="scene-parse-150",
...     eval_accumulation_steps=5,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=train_ds,
...     eval_dataset=test_ds,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

### انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

ڈیٹاسیٹ کو دوبارہ لوڈ کریں اور انفرنس کے لیے ایک تصویر لوڈ کریں۔

```py
>>> from datasets import load_dataset

>>> ds = load_dataset("merve/scene_parse_150", split="train[:50]")
>>> ds = ds.train_test_split(test_size=0.2)
>>> test_ds = ds["test"]
>>> image = ds["test"][0]["image"]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-image.png" alt="Image of bedroom"/>
</div>

اب ہم دیکھیں گے کہ پائپ لائن کے بغیر کیسے اندازہ لگایا جائے۔ امیج پروسیسر کے ساتھ تصویر پر کارروائی کریں اور `pixel_values` کو GPU پر رکھیں:

```py
>>> from accelerate import Accelerator
>>> device = Accelerator().device
>>> encoding = image_processor(image, return_tensors="pt")
>>> pixel_values = encoding.pixel_values.to(device)
```

اپنا ان پٹ ماڈل پر دیں اور `logits` واپس کریں:

```py
>>> outputs = model(pixel_values=pixel_values)
>>> logits = outputs.logits.cpu()
```

اگلا، لاگٹس کو اصل تصویر کے سائز پر ری سکیل کریں:

```py
>>> upsampled_logits = nn.functional.interpolate(
...     logits,
...     size=image.size[::-1],
...     mode="bilinear",
...     align_corners=False,
... )

>>> pred_seg = upsampled_logits.argmax(dim=1)[0]
```

نتائج کو دیکھنے کے لیے، [ڈیٹا سیٹ کلر پیلیٹ](https://github.com/tensorflow/models/blob/3f1ca33afe3c1631b733ea7e40c294273b9e406d/research/deeplab/utils/get_dataset_colormap.py#L51) کو `ade_palette()` کے طور پر لوڈ کریں جو ہر کلاس کو ان کی RGB اقدار کے مطابق بناتا ہے۔

```py
def ade_palette():
  return np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])
```

اس کے بعد آپ اپنی تصویر اور پیش گوئی شدہ سیگمنٹیشن میپ کو جوڑ کر پلاٹ کر سکتے ہیں:

```py
>>> import matplotlib.pyplot as plt
>>> import numpy as np

>>> color_seg = np.zeros((pred_seg.shape[0], pred_seg.shape[1], 3), dtype=np.uint8)
>>> palette = np.array(ade_palette())
>>> for label, color in enumerate(palette):
...     color_seg[pred_seg == label, :] = color
>>> color_seg = color_seg[..., ::-1]  # convert to BGR

>>> img = np.array(image) * 0.5 + color_seg * 0.5  # plot the image with the segmentation map
>>> img = img.astype(np.uint8)

>>> plt.figure(figsize=(15, 10))
>>> plt.imshow(img)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/semantic-seg-preds.png" alt="Image of bedroom overlaid with segmentation map"/>
</div>
