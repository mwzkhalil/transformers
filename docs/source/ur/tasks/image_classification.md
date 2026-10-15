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

# تصویر کی درجہ بندی

[[open-in-colab]]

<Youtube id="tjAIM7BOYhw"/>

تصویر کی درجہ بندی کسی تصویر کو ایک لیبل یا کلاس تفویض کرتی ہے۔ متن یا آڈیو کی درجہ بندی کے برعکس، ان پٹ پکسل کی اقدار ہیں جو ایک تصویر پر مشتمل ہوتی ہیں۔ تصویر کی درجہ بندی کے لیے بہت سی ایپلی کیشنز ہیں، جیسے کہ قدرتی آفت کے بعد نقصان کا پتہ لگانا، فصل کی صحت کی نگرانی کرنا، یا بیماری کی علامات کے لیے طبی امیجوں کی اسکرین میں مدد کرنا۔

یہ گائیڈ وضاحت کرتا ہے کہ کیسے:

1. فائن ٹیون [ViT](https://huggingface.co/docs/transformers/main/en/model_doc/vit) [Food-101](https://huggingface.co/datasets/ethz/food101) ڈیٹاسیٹ پر ایک تصویر میں کھانے کی اشیاء کی درجہ بندی کرنے کے لیے۔
2. انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/image-classification) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate accelerate pillow torchvision scikit-learn trackio
```

ہم آپ کی حوصلہ افزائی کرتے ہیں کہ آپ اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں تاکہ اپنا ماڈل اپ لوڈ کریں اور کمیونٹی کے ساتھ شیئر کریں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## Food-101 ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے Food-101 ڈیٹاسیٹ کا ایک چھوٹا سب سیٹ لوڈ کرکے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from datasets import load_dataset

>>> food = load_dataset("ethz/food101", split="train[:5000]")
```

ڈیٹا سیٹ کے `train` کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```py
>>> food = food.train_test_split(test_size=0.2)
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> food["train"][0]
{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F52AFC8AC50>,
 'label': 79}
```

ڈیٹاسیٹ میں ہر مثال کے دو فیلڈز ہوتے ہیں:

- `image`: کھانے کی اشیاء کی PIL تصویر
- `label`: کھانے کی اشیاء کی لیبل کلاس

ماڈل کے لیے لیبل آئی ڈی سے لیبل کا نام حاصل کرنا آسان بنانے کے لیے، ایک لغت بنائیں جو لیبل کے نام کو ایک عدد میں نقشے اور اس کے برعکس:

```py
>>> labels = food["train"].features["label"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

اب آپ لیبل آئی ڈی کو لیبل کے نام میں تبدیل کر سکتے ہیں:

```py
>>> id2label[str(79)]
'prime_rib'
```

## پری پروسیس

اگلا مرحلہ تصویر کو ٹینسر میں پروسیس کرنے کے لیے ViT امیج پروسیسر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoImageProcessor

>>> checkpoint = "google/vit-base-patch16-224-in21k"
>>> image_processor = AutoImageProcessor.from_pretrained(checkpoint)
```

اوور فٹنگ کے خلاف ماڈل کو مزید مضبوط بنانے کے لیے تصاویر میں کچھ تصویری تبدیلیاں لگائیں۔ یہاں آپ ٹارچ ویژن کا [`transforms`](https://pytorch.org/vision/stable/transforms.html) ماڈیول استعمال کریں گے، لیکن آپ اپنی پسند کی کوئی بھی تصویری لائبریری بھی استعمال کر سکتے ہیں۔

تصویر کے بے ترتیب حصے کو تراشیں، اس کا سائز تبدیل کریں، اور تصویر کے وسط اور معیاری انحراف کے ساتھ اسے معمول پر لائیں:

```py
>>> from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

>>> normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
>>> size = (
...     image_processor.size["shortest_edge"]
...     if "shortest_edge" in image_processor.size
...     else (image_processor.size["height"], image_processor.size["width"])
... )
>>> _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])
```

پھر ٹرانسفارمز کو لاگو کرنے کے لیے ایک پری پروسیسنگ فنکشن بنائیں اور `pixel_values` - تصویر کے ماڈل میں ان پٹ واپس کریں:

```py
>>> def transforms(examples):
...     examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
...     del examples["image"]
...     return examples
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.with_transform`] طریقہ استعمال کریں۔ جب آپ ڈیٹاسیٹ کا ایک عنصر لوڈ کرتے ہیں تو تبدیلیوں کو فلائی پر لاگو کیا جاتا ہے:

```py
>>> food = food.with_transform(transforms)
```

اب [`DefaultDataCollator`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ 🤗 Transformers میں دیگر ڈیٹا کولیٹرز کے برعکس، `DefaultDataCollator` اضافی پری پروسیسنگ جیسے پیڈنگ کا اطلاق نہیں کرتا ہے۔

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) میٹرک لوڈ کریں (میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے طریقے کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

پھر ایک فنکشن بنائیں جو درستگی کا حساب لگانے کے لیے آپ کی پیشین گوئیوں اور لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ ترتیب دیں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForImageClassification`] کے ساتھ ViT لوڈ کریں۔ متوقع لیبلز کی تعداد کے ساتھ لیبل کی تعداد اور لیبل میپنگ کی وضاحت کریں:

```py
>>> from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

>>> model = AutoModelForImageClassification.from_pretrained(
...     checkpoint,
...     num_labels=len(labels),
...     id2label=id2label,
...     label2id=label2id,
... )
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ یہ ضروری ہے کہ آپ غیر استعمال شدہ کالموں کو نہ ہٹائیں کیونکہ اس سے `image` کالم گر جائے گا۔ `image` کالم کے بغیر، آپ `pixel_values` نہیں بنا سکتے۔ اس رویے کو روکنے کے لیے `remove_unused_columns=False` سیٹ کریں! صرف دوسرا مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] درستگی کا اندازہ کرے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_food_model",
...     remove_unused_columns=False,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     warmup_steps=0.1,
...     logging_steps=10,
...     report_to="trackio",
...     run_name="food101",
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=food["train"],
...     eval_dataset=food["test"],
...     processing_class=image_processor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

تصویر کی درجہ بندی کے لیے ماڈل کو ٹھیک کرنے کے طریقے کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/image_classification.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ایک ماڈل ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

ایک تصویر لوڈ کریں جس پر آپ انفرنس چلانا چاہتے ہیں:

```py
>>> ds = load_dataset("ethz/food101", split="validation[:10]")
>>> image = ds["image"][0]
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png" alt="image of beignets"/>
</div>

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ تصویر کی درجہ بندی کے لیے ایک `pipeline` فوری بنائیں، اور اپنی تصویر اس کو منتقل کریں:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("image-classification", model="my_awesome_food_model")
>>> classifier(image)
[{'score': 0.31856709718704224, 'label': 'beignets'},
 {'score': 0.015232225880026817, 'label': 'bruschetta'},
 {'score': 0.01519392803311348, 'label': 'chicken_wings'},
 {'score': 0.013022331520915031, 'label': 'pork_chop'},
 {'score': 0.012728818692266941, 'label': 'prime_rib'}]
```

اگر آپ چاہیں تو آپ `pipeline` کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں:

امیج کو پری پروسیس کرنے کے لیے ایک امیج پروسیسر لوڈ کریں اور `input` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoImageProcessor
>>> import torch

>>> image_processor = AutoImageProcessor.from_pretrained("my_awesome_food_model")
>>> inputs = image_processor(image, return_tensors="pt")
```

اپنے ان پٹس کو ماڈل پر منتقل کریں اور لاگٹس واپس کریں:

```py
>>> from transformers import AutoModelForImageClassification

>>> model = AutoModelForImageClassification.from_pretrained("my_awesome_food_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

سب سے زیادہ امکان کے ساتھ پیشین گوئی شدہ لیبل حاصل کریں، اور اسے لیبل میں تبدیل کرنے کے لیے ماڈل کی `id2label` میپنگ کا استعمال کریں:

```py
>>> predicted_label = logits.argmax(-1).item()
>>> model.config.id2label[predicted_label]
'beignets'
```
