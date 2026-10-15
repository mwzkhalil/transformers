<!---
Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# خرابی کا سراغ لگانا

بعض اوقات غلطیاں ہوتی ہیں، لیکن ہم مدد کے لیے حاضر ہیں! یہ گائیڈ کچھ عام مسائل کا احاطہ کرتا ہے جو ہم نے دیکھے ہیں اور آپ انہیں کیسے حل کر سکتے ہیں۔ تاہم، اس گائیڈ کا مقصد ہر 🤗 Transformers شمارے کا ایک جامع مجموعہ نہیں ہے۔ اپنے مسئلے کو حل کرنے میں مزید مدد کے لیے، کوشش کریں:

<Youtube id="S2EEG3JIt2A"/>

1. [فورمز](https://discuss.huggingface.co/) پر مدد طلب کرنا۔ کچھ مخصوص زمرے ہیں جن پر آپ اپنا سوال پوسٹ کر سکتے ہیں، جیسے [Beginners](https://discuss.huggingface.co/c/beginners/5) یا [🤗 Transformers](https://discuss.huggingface.co/c/transformers/9)۔ اس بات کو یقینی بنائیں کہ آپ نے ایک اچھی وضاحتی فورم پوسٹ لکھی ہے جس میں کچھ تولیدی کوڈ ہیں تاکہ آپ کا مسئلہ حل ہونے کا امکان زیادہ سے زیادہ ہو!

<Youtube id="_PAli-V4wj0"/>

2. 🤗 Transformers ریپوزٹری پر ایک [مسئلہ](https://github.com/huggingface/transformers/issues/new/choose) بنائیں اگر یہ لائبریری سے متعلق کوئی بگ ہے۔ غلطی کی وضاحت کرنے والی زیادہ سے زیادہ معلومات شامل کرنے کی کوشش کریں تاکہ ہمیں بہتر طریقے سے یہ معلوم کرنے میں مدد ملے کہ کیا غلط ہے اور ہم اسے کیسے ٹھیک کر سکتے ہیں۔

3. اگر آپ 🤗 Transformers کا پرانا ورژن استعمال کرتے ہیں تو [Migration](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md) گائیڈ کو چیک کریں کیونکہ ورژن کے درمیان کچھ اہم تبدیلیاں متعارف کرائی گئی ہیں۔

ٹربل شوٹنگ اور مدد حاصل کرنے کے بارے میں مزید تفصیلات کے لیے، [Hugging Face کورس کے باب 8](https://huggingface.co/course/chapter8/1?fw=pt) پر ایک نظر ڈالیں۔

## فائر وال والے ماحول

کلاؤڈ اور انٹرانیٹ سیٹ اپ پر کچھ GPU مثالیں بیرونی کنکشنز پر فائر والڈ ہیں، جس کے نتیجے میں کنکشن کی خرابی ہوتی ہے۔ جب آپ کا اسکرپٹ ماڈل وزن یا ڈیٹا سیٹ ڈاؤن لوڈ کرنے کی کوشش کرتا ہے، تو ڈاؤن لوڈ لٹک جائے گا اور پھر درج ذیل پیغام کے ساتھ ٹائم آؤٹ ہو جائے گا:

```text
ValueError: Connection error, and we cannot find the requested files in the cached path.
Please try again or make sure your Internet connection is on.
```

اس صورت میں، آپ کو کنکشن کی خرابی سے بچنے کے لیے 🤗 Transformers کو [آف لائن موڈ](installation#offline-mode) پر چلانے کی کوشش کرنی چاہیے۔

## CUDA میموری سے باہر ہے۔

لاکھوں پیرامیٹرز کے ساتھ بڑی ماڈلز تربیت کرنا مناسب ہارڈ ویئر کے بغیر مشکل ہو سکتا ہے۔ GPU میموری ختم ہونے پر آپ کو ایک عام غلطی کا سامنا کرنا پڑ سکتا ہے:

```text
CUDA out of memory. Tried to allocate 256.00 MiB (GPU 0; 11.17 GiB total capacity; 9.70 GiB already allocated; 179.81 MiB free; 9.85 GiB reserved in total by PyTorch)
```

یہاں کچھ ممکنہ حل ہیں جو آپ میموری کے استعمال کو کم کرنے کی کوشش کر سکتے ہیں:

- [`~TrainingArguments#per_device_train_batch_size`] قدر کو [`TrainingArguments`] میں کم کریں۔
- مجموعی طور پر بیچ کے سائز کو مؤثر طریقے سے بڑھانے کے لیے [`TrainingArguments`] میں [`~TrainingArguments#gradient_accumulation_steps`] استعمال کرنے کی کوشش کریں۔

<Tip>

یادداشت بچانے کی تکنیکوں کے بارے میں مزید تفصیلات کے لیے [آپٹمائزیشن کا جائزہ](https://huggingface.co/docs/transformers/main/en/optimization_overview) دیکھیں۔

</Tip>

## درآمد کی خرابی۔

ایک اور عام غلطی جس کا آپ کو سامنا ہو سکتا ہے، خاص طور پر اگر یہ ایک نیا جاری کردہ ماڈل ہے، ہے `ImportError`:

```text
ImportError: cannot import name 'ImageGPTImageProcessor' from 'transformers' (unknown location)
```

خرابی کی ان اقسام کے لیے، یہ یقینی بنانے کے لیے چیک کریں کہ آپ کے پاس 🤗 Transformers کا تازہ ترین ورژن انسٹال ہے تاکہ تازہ ترین ماڈلز تک رسائی حاصل کی جا سکے۔

```bash
pip install transformers --upgrade
```

## CUDA کی خرابی: ڈیوائس سائیڈ ایسٹ ٹرگر ہو گیا۔

کبھی کبھی آپ کو ڈیوائس کوڈ میں خرابی کے بارے میں ایک عام CUDA غلطی کا سامنا کرنا پڑ سکتا ہے۔

```text
RuntimeError: CUDA error: device-side assert triggered
```

مزید وضاحتی غلطی کا پیغام حاصل کرنے کے لیے آپ کو پہلے CPU پر کوڈ چلانے کی کوشش کرنی چاہیے۔ سی پی یو پر سوئچ کرنے کے لیے اپنے کوڈ کے آغاز میں درج ذیل ماحولیاتی متغیر کو شامل کریں:

```py
>>> import os

>>> os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

دوسرا آپشن GPU سے بہتر ٹریس بیک حاصل کرنا ہے۔ غلطی کے ماخذ کی طرف اشارہ کرنے کے لیے ٹریس بیک حاصل کرنے کے لیے اپنے کوڈ کے آغاز میں درج ذیل ماحولیاتی متغیر کو شامل کریں:

```py
>>> import os

>>> os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
```

## جب پیڈنگ ٹوکنز کو ماسک نہیں کیا جاتا ہے تو غلط آؤٹ پٹ

کچھ صورتوں میں، آؤٹ پٹ `hidden_state` غلط ہو سکتا ہے اگر `input_ids` میں پیڈنگ ٹوکنز شامل ہوں۔ ظاہر کرنے کے لیے، ایک ماڈل اور ٹوکنائزر لوڈ کریں۔ آپ اس کی قدر دیکھنے کے لیے ماڈل's `pad_token_id` تک رسائی حاصل کر سکتے ہیں۔ `pad_token_id` کچھ ماڈلز کے لیے `None` ہو سکتا ہے، لیکن آپ اسے ہمیشہ دستی طور پر سیٹ کر سکتے ہیں۔

```py
>>> from transformers import AutoModelForSequenceClassification
>>> import torch

>>> model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
>>> model.config.pad_token_id
0
```

مندرجہ ذیل مثال پیڈنگ ٹوکن کو ماسک کیے بغیر آؤٹ پٹ دکھاتی ہے۔

```py
>>> input_ids = torch.tensor([[7592, 2057, 2097, 2393, 9611, 2115], [7592, 0, 0, 0, 0, 0]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [ 0.1317, -0.1683]], grad_fn=<AddmmBackward0>)
```

یہاں دوسری ترتیب کی اصل پیداوار ہے:

```py
>>> input_ids = torch.tensor([[7592]])
>>> output = model(input_ids)
>>> print(output.logits)
tensor([[-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

زیادہ تر وقت، آپ کو اس خاموش غلطی سے بچنے کے لیے پیڈنگ ٹوکنز کو نظر انداز کرنے کے لیے اپنے ماڈل کو `attention_mask` فراہم کرنا چاہیے۔ اب دوسری ترتیب کا آؤٹ پٹ اس کے اصل آؤٹ پٹ سے میل کھاتا ہے:

<Tip>

پہلے سے طے شدہ طور پر، ٹوکنائزر آپ کے مخصوص ٹوکنائزر کے ڈیفالٹس کی بنیاد پر آپ کے لیے ایک `attention_mask` بناتا ہے۔

</Tip>

```py
>>> attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 0]])
>>> output = model(input_ids, attention_mask=attention_mask)
>>> print(output.logits)
tensor([[ 0.0082, -0.2307],
        [-0.1008, -0.4061]], grad_fn=<AddmmBackward0>)
```

🤗 Transformers پیڈنگ ٹوکن کو ماسک کرنے کے لیے خود بخود `attention_mask` نہیں بناتا ہے اگر یہ فراہم کیا جاتا ہے کیونکہ:

- کچھ ماڈلز میں پیڈنگ ٹوکن نہیں ہے۔
- استعمال کے کچھ معاملات میں، صارفین پیڈنگ ٹوکن میں شرکت کے لیے ماڈل چاہتے ہیں۔

## ویلیو ایرر: اس قسم کے آٹو ماڈل کے لیے غیر تسلیم شدہ کنفیگریشن کلاس XYZ

عام طور پر، ہم ماڈلز کی پہلے سے تربیت یافتہ مثالوں کو لوڈ کرنے کے لیے [`AutoModel`] کلاس استعمال کرنے کی تجویز کرتے ہیں۔ یہ کلاس کنفیگریشن کی بنیاد پر دیے گئے چیک پوائنٹ سے درست فن تعمیر کا خود بخود اندازہ اور لوڈ کر سکتی ہے۔ اگر آپ کو چیک پوائنٹ سے ماڈل لوڈ کرتے وقت یہ `ValueError` نظر آتا ہے، تو اس کا مطلب ہے کہ آٹو کلاس کو دی گئی چیک پوائنٹ میں کنفیگریشن سے اس قسم کی میپنگ نہیں مل سکی جس کی آپ ماڈل لوڈ کرنے کی کوشش کر رہے ہیں۔ عام طور پر، ایسا اس وقت ہوتا ہے جب چیک پوائنٹ دیئے گئے کام کی حمایت نہیں کرتا ہے۔ مثال کے طور پر، آپ کو یہ غلطی درج ذیل مثال میں نظر آئے گی کیونکہ سوال کے جواب کے لیے کوئی GPT2 نہیں ہے:

```py
>>> from transformers import AutoProcessor, AutoModelForQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("openai-community/gpt2-medium")
>>> model = AutoModelForQuestionAnswering.from_pretrained("openai-community/gpt2-medium")
ValueError: Unrecognized configuration class <class 'transformers.models.gpt2.configuration_gpt2.GPT2Config'> for this kind of AutoModel: AutoModelForQuestionAnswering.
Model type should be one of AlbertConfig, BartConfig, BertConfig, BigBirdConfig, BigBirdPegasusConfig, BloomConfig, ...
```
