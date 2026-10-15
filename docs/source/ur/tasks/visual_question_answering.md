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

# بصری سوال جواب

[[open-in-colab]]

بصری سوال جواب (VQA) ایک تصویر کی بنیاد پر کھلے سوالات کے جوابات دینے کا کام ہے۔ ماڈلز اس کام کو سپورٹ کرنے کا ان پٹ عام طور پر ایک تصویر اور سوال کا مجموعہ ہوتا ہے، اور آؤٹ پٹ ایک جواب ہوتا ہے جس کا اظہار قدرتی زبان میں ہوتا ہے۔

VQA کے لیے استعمال کی کچھ قابل ذکر مثالیں شامل ہیں:

* نابینا افراد کے لیے قابل رسائی ایپلی کیشنز۔
* تعلیم: لیکچرز یا درسی کتابوں میں پیش کردہ بصری مواد کے بارے میں سوالات کرنا۔ VQA کو انٹرایکٹو میوزیم کی نمائشوں یا تاریخی مقامات میں بھی استعمال کیا جا سکتا ہے۔
* کسٹمر سروس اور ای کامرس: VQA صارفین کو مصنوعات کے بارے میں سوالات پوچھنے کی اجازت دے کر صارف کے تجربے کو بڑھا سکتا ہے۔
* تصویر کی بازیافت: VQA ماڈلز کو مخصوص خصوصیات کے ساتھ تصاویر کی بازیافت کے لیے استعمال کیا جا سکتا ہے۔ مثال کے طور پر، صارف پوچھ سکتا ہے "کیا کوئی کتا ہے؟" تصاویر کے سیٹ سے کتوں کے ساتھ تمام تصاویر تلاش کرنے کے لیے۔

اس گائیڈ میں آپ یہ سیکھیں گے کہ:

- فائن ٹیون درجہ بندی VQA ماڈل، خاص طور پر [ViLT](https://huggingface.co/docs/transformers/main/en/model_doc/vilt)، [`Graphcore/vqa` ڈیٹا سیٹ](https://huggingface.co/datasets/Graphcore/vqa) پر۔
- انفرنس کے لیے اپنا فائن ٹیونڈ ViLT استعمال کریں۔
- زیرو شاٹ VQA انفرنس کو جنریٹو ماڈل کے ساتھ چلائیں، جیسے BLIP-2۔

## فائن ٹیوننگ ViLT

ViLT ماڈل ٹیکسٹ ایمبیڈنگز کو وژن ٹرانسفارمر (ViT) میں شامل کرتا ہے، جس سے اسے Vision-and-Language Pre-Training (VLP) کے لیے کم سے کم ڈیزائن مل سکتا ہے۔ یہ ماڈل کئی بہاو والے کاموں کے لیے استعمال کیا جا سکتا ہے۔ VQA کام کے لیے، ایک درجہ بندی کا سر اوپر رکھا جاتا ہے (`[CLS]` ٹوکن کی آخری پوشیدہ حالت کے اوپر ایک لکیری پرت) اور تصادفی طور پر شروع کیا جاتا ہے۔ اس طرح بصری سوال کے جواب کو **درجہ بندی کا مسئلہ** سمجھا جاتا ہے۔

تازہ ترین ماڈلز، جیسے BLIP، BLIP-2، اور InstructBLIP، VQA کو ایک تخلیقی کام کے طور پر مانتے ہیں۔ بعد میں اس گائیڈ میں ہم وضاحت کرتے ہیں کہ انہیں صفر شاٹ VQA انفرنس کے لیے کیسے استعمال کیا جائے۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں۔

```bash
pip install -q transformers datasets
```

ہم آپ کی حوصلہ افزائی کرتے ہیں کہ اپنا ماڈل کمیونٹی کے ساتھ شیئر کریں۔ اسے 🤗 حب پر اپ لوڈ کرنے کے لیے اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

آئیے ماڈل چیک پوائنٹ کو عالمی متغیر کے طور پر بیان کرتے ہیں۔

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
```

## ڈیٹا لوڈ کریں۔

مثال کے مقاصد کے لیے، اس گائیڈ میں ہم تشریح شدہ بصری سوال کا جواب دینے والے `Graphcore/vqa` ڈیٹا سیٹ کا ایک بہت چھوٹا نمونہ استعمال کرتے ہیں۔ آپ مکمل ڈیٹا سیٹ [🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa) پر حاصل کر سکتے ہیں۔

[`Graphcore/vqa` dataset](https://huggingface.co/datasets/Graphcore/vqa) کے متبادل کے طور پر، آپ وہی ڈیٹا آفیشل [VQA ڈیٹاسیٹ صفحہ](https://visualqa.org/download.html) سے دستی طور پر ڈاؤن لوڈ کر سکتے ہیں۔ اگر آپ اپنے حسب ضرورت ڈیٹا کے ساتھ ٹیوٹوریل کی پیروی کرنا پسند کرتے ہیں تو، 🤗 ڈیٹاسیٹس کی دستاویزات میں [تصویری ڈیٹاسیٹ](https://huggingface.co/docs/datasets/image_dataset#loading-script) گائیڈ بنانے کا طریقہ دیکھیں۔

آئیے توثیق کی تقسیم سے پہلی 200 مثالیں لوڈ کریں اور ڈیٹا سیٹ کی خصوصیات کو دریافت کریں:

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
    features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
    num_rows: 200
})
```

آئیے ڈیٹا سیٹ کی خصوصیات کو سمجھنے کے لیے ایک مثال پر ایک نظر ڈالتے ہیں:

```py
>>> dataset[0]
{'question': 'Where is he looking?',
 'question_type': 'none of the above',
 'question_id': 262148000,
 'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
 'answer_type': 'other',
 'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
  'weights': [0.30000001192092896,
   1.0,
   0.30000001192092896,
   0.30000001192092896]}}
```

کام سے متعلقہ خصوصیات میں شامل ہیں:

* `question`: تصویر سے جس سوال کا جواب دینا ہے۔
* `image_id`: تصویر کا راستہ جس سے سوال مراد ہے۔
* `label`: تشریحات

ہم باقی خصوصیات کو ہٹا سکتے ہیں کیونکہ ان کی ضرورت نہیں ہوگی:

```py
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
```

جیسا کہ آپ دیکھ سکتے ہیں، `label` خصوصیت میں ایک ہی سوال کے متعدد جوابات ہیں (جسے یہاں `ids` کہا جاتا ہے) مختلف انسانی تشریح کاروں کے ذریعے جمع کیا گیا ہے۔ اس کی وجہ یہ ہے کہ سوال کا جواب موضوعی ہو سکتا ہے۔ اس معاملے میں، سوال یہ ہے کہ "وہ کہاں دیکھ رہا ہے؟"۔ کچھ لوگوں نے اسے "نیچے"، دوسروں نے "ٹیبل پر"، دوسرے نے "سکیٹ بورڈ" وغیرہ کے ساتھ تشریح کی۔

تصویر پر ایک نظر ڈالیں اور غور کریں کہ آپ کون سا جواب دیں گے:

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
```

<div class="flex justify-center">
     <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div>

سوالات اور جوابات کے ابہام کی وجہ سے، اس طرح کے ڈیٹاسیٹس کو ملٹی لیبل درجہ بندی کا مسئلہ سمجھا جاتا ہے (کیونکہ متعدد جوابات ممکنہ طور پر درست ہیں)۔ مزید یہ کہ، صرف ایک گرم انکوڈ شدہ ویکٹر بنانے کے بجائے، کوئی ایک نرم انکوڈنگ بناتا ہے، جس کی بنیاد پر تشریحات میں کوئی خاص جواب کتنی بار ظاہر ہوتا ہے۔

مثال کے طور پر، اوپر دی گئی مثال میں، چونکہ جواب "نیچے" کو دوسرے جوابات کے مقابلے زیادہ کثرت سے منتخب کیا جاتا ہے، اس لیے اس کا اسکور (ڈیٹا سیٹ میں `weight` کہا جاتا ہے) 1.0 ہے، اور باقی جوابات کے اسکور <1.0 ہیں۔

بعد میں ماڈل کو ایک مناسب درجہ بندی کے سر کے ساتھ انسٹیٹیوٹ کرنے کے لیے، آئیے دو لغات بنائیں: ایک جو لیبل کے نام کو عدد پر نقشہ بناتا ہے اور اس کے برعکس:

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()}
```

اب جب کہ ہمارے پاس میپنگز ہیں، ہم سٹرنگ جوابات کو ان کے آئی ڈی سے بدل سکتے ہیں، اور مزید آسان مزید پری پروسیسنگ کے لیے ڈیٹاسیٹ کو فلیٹ کر سکتے ہیں۔

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
 'image_id': Value(dtype='string', id=None),
 'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
 'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
```

## ڈیٹا پری پروسیسنگ

اگلا مرحلہ ماڈل کے لیے امیج اور ٹیکسٹ ڈیٹا تیار کرنے کے لیے ViLT پروسیسر لوڈ کرنا ہے۔ [`ViltProcessor`] BERT ٹوکنائزر اور ViLT امیج پروسیسر کو ایک آسان سنگل پروسیسر میں لپیٹتا ہے:

```py
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
```

ڈیٹا کو پری پروسیس کرنے کے لیے ہمیں [`ViltProcessor`] کا استعمال کرتے ہوئے تصاویر اور سوالات کو انکوڈ کرنے کی ضرورت ہے۔ پروسیسر متن کو ٹوکنائز کرنے کے لیے [`BertTokenizerFast`] کا استعمال کرے گا اور ٹیکسٹ ڈیٹا کے لیے `input_ids`، `attention_mask` اور `token_type_ids` بنائے گا۔ جہاں تک تصویروں کا تعلق ہے، پروسیسر تصویر کا سائز تبدیل کرنے اور معمول پر لانے کے لیے [`ViltImageProcessor`] کا فائدہ اٹھائے گا، اور `pixel_values` اور `pixel_mask` بنائے گا۔

یہ تمام پری پروسیسنگ اقدامات ہڈ کے نیچے کیے جاتے ہیں، ہمیں صرف `processor` کال کرنے کی ضرورت ہے۔ تاہم، ہمیں ابھی بھی ہدف کے لیبل تیار کرنے کی ضرورت ہے۔ اس نمائندگی میں، ہر عنصر ممکنہ جواب (لیبل) سے مطابقت رکھتا ہے۔ درست جوابات کے لیے، عنصر اپنے متعلقہ سکور (وزن) رکھتا ہے، جبکہ باقی عناصر صفر پر سیٹ ہوتے ہیں۔

مندرجہ ذیل فنکشن `processor` کو تصاویر اور سوالات پر لاگو کرتا ہے اور لیبل کو فارمیٹ کرتا ہے جیسا کہ اوپر بیان کیا گیا ہے:

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()

...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score

...         targets.append(target)

...     encoding["labels"] = targets

...     return encoding
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.map`] فنکشن استعمال کریں۔ آپ `map` سیٹ کر کے `batched=True` ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پراسیس کر سکتے ہیں۔ اس وقت، بلا جھجھک ان کالموں کو ہٹا دیں جن کی آپ کو ضرورت نہیں ہے۔

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
    features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
    num_rows: 200
})
```

حتمی قدم کے طور پر، [`DefaultDataCollator`] استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں:

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## ماڈل کو تربیت دیں

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`ViltForQuestionAnswering`] کے ساتھ ViLT لوڈ کریں۔ لیبل میپنگ کے ساتھ لیبلز کی تعداد کی وضاحت کریں:

```py
>>> from transformers import ViltForQuestionAnswering

>>> model = ViltForQuestionAnswering.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں:

```py
>>> from transformers import TrainingArguments

>>> repo_id = "MariaK/vilt_finetuned_200"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```

2. ماڈل، ڈیٹا سیٹ، پروسیسر، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل کو [`Trainer`] تک منتقل کریں۔

```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=processed_dataset,
...     processing_class=processor,
... )
```

3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> trainer.train()
```

ٹریننگ مکمل ہونے کے بعد، اپنے ماڈل کو [`~Trainer.push_to_hub`] طریقہ کے ساتھ حب میں شیئر کریں تاکہ 🤗 Hub پر اپنا حتمی ماڈل شیئر کریں:

```py
>>> trainer.push_to_hub()
```

## انفرنس

اب جب کہ آپ نے ایک ViLT ماڈل کو ٹھیک کر لیا ہے، اور اسے 🤗 Hub پر اپ لوڈ کر دیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں۔

1. ایک تصویر اور سوال لیں، انہیں اپنے ماڈل سے پروسیسر کا استعمال کرتے ہوئے ماڈل کے لیے تیار کریں۔
2. نتیجہ آگے بھیجیں یا ماڈل کے ذریعے پری پروسیسنگ کریں۔
3. لاگٹس سے، ممکنہ جواب کی آئی ڈی حاصل کریں، اور `id2label` میں اصل جواب تلاش کریں۔

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # prepare inputs
>>> inputs = processor(image, question, return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # forward pass
>>> with torch.no_grad():
...     outputs = model(**inputs)

>>> logits = outputs.logits
>>> idx = logits.argmax(-1).item()
>>> print("Predicted answer:", model.config.id2label[idx])
Predicted answer: down
```

## زیرو شاٹ VQA

پچھلے ماڈل نے VQA کو درجہ بندی کے کام کے طور پر سمجھا تھا۔ کچھ حالیہ ماڈلز، جیسے BLIP، BLIP-2، اور InstructBLIP VQA کو تخلیقی کام کے طور پر اپروچ کرتے ہیں۔ آئیے ایک مثال کے طور پر [BLIP-2](https://huggingface.co/docs/transformers/main/en/model_doc/blip-2) لیتے ہیں۔ اس نے بصری زبان سے پہلے کی تربیت کا ایک نیا نمونہ متعارف کرایا جس میں پہلے سے تربیت یافتہ وژن انکوڈر اور LLM کا کوئی بھی مجموعہ استعمال کیا جا سکتا ہے (مزید جانیں [BLIP-2 بلاگ پوسٹ](https://huggingface.co/blog/blip-2) میں)۔ یہ بصری سوالوں کے جوابات سمیت متعدد بصری زبان کے کاموں پر جدید ترین نتائج حاصل کرنے کے قابل بناتا ہے۔

آئیے وضاحت کرتے ہیں کہ آپ VQA ​​کے لیے اس ماڈل کو کس طرح استعمال کر سکتے ہیں۔ سب سے پہلے، آئیے ماڈل لوڈ کرتے ہیں۔ یہاں ہم واضح طور پر ماڈل ایک GPU کو بھیجیں گے، اگر دستیاب ہو، جو ہمیں تربیت کے دوران پہلے کرنے کی ضرورت نہیں تھی، کیونکہ [`Trainer`] اسے خود بخود ہینڈل کرتا ہے:

```py
>>> from transformers import AutoProcessor, Blip2ForConditionalGeneration
from accelerate import Accelerator
>>> import torch

>>> processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16)
>>> device = Accelerator().device
>>> model.to(device)
```

ماڈل تصویر اور متن کو بطور ان پٹ لیتا ہے، تو آئیے VQA ڈیٹاسیٹ میں پہلی مثال سے بالکل وہی تصویر/سوال کا جوڑا استعمال کریں:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

بصری سوال کے جواب دینے کے کام کے لیے BLIP-2 استعمال کرنے کے لیے، ٹیکسٹول پرامپٹ کو ایک مخصوص فارمیٹ پر عمل کرنا ہوگا: `Question: {} Answer:`۔

```py
>>> prompt = f"Question: {question} Answer:"
```

اب ہمیں ماڈل کے پروسیسر کے ساتھ امیج/پرامپٹ کو پری پروسیس کرنے کی ضرورت ہے، پروسیس شدہ ان پٹ کو ماڈل کے ذریعے پاس کریں، اور آؤٹ پٹ کو ڈی کوڈ کریں:

```py
>>> inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)

>>> generated_ids = model.generate(**inputs, max_new_tokens=10)
>>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
>>> print(generated_text)
"He is looking at the crowd"
```

جیسا کہ آپ دیکھ سکتے ہیں، ماڈل نے ہجوم کو پہچان لیا، اور چہرے کی سمت (نیچے دیکھتے ہوئے)، تاہم، ایسا لگتا ہے کہ ہجوم اسکیٹر کے پیچھے ہے۔ پھر بھی، ایسے معاملات میں جہاں انسانی تشریح شدہ ڈیٹاسیٹس کا حصول ممکن نہیں ہے، یہ نقطہ نظر تیزی سے مفید نتائج پیدا کر سکتا ہے۔
