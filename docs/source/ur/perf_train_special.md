<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Apple Silicon

Apple Silicon (M-series) چپس میں ایک متحد میموری فن تعمیر ہے جہاں CPU اور GPU ایک ہی میموری پول کا اشتراک کرتے ہیں۔ مشترکہ میموری GPUs کے ڈیٹا کی منتقلی کو ختم کر دیتی ہے، جس سے مقامی طور پر بڑے ماڈلز کو تربیت دینا عملی ہو جاتا ہے۔ Transformers اس ہارڈ ویئر پر [میٹل پرفارمنس شیڈرز (MPS)](https://pytorch.org/docs/stable/notes/mps.html) بیک اینڈ کو Accelerate ٹریننگ کا استعمال کرتا ہے۔

اس کے لیے macOS 12.3 یا بعد کا اور MPS سپورٹ کے ساتھ بنایا ہوا PyTorch درکار ہے۔

> [!انتباہ] > MPS ابھی تک تمام PyTorch آپریشنز کو سپورٹ نہیں کرتا ہے (یہ دیکھیں [GitHub issue](https://github.com/pytorch/pytorch/issues/77764) لاپتہ آپریشنز کے بارے میں مزید تفصیلات کے لیے)۔ سیٹ `PYTORCH_ENABLE_MPS_FALLBACK=1` غیر تعاون یافتہ آپریشنز کے لیے CPU کرنل پر واپس آنے کے لیے۔ کسی دوسرے غیر متوقع رویے کے لیے [PyTorch](https://github.com/pytorch/pytorch/issues) ذخیرہ میں ایک مسئلہ کھولیں۔



## ماڈل لوڈنگ اور ڈیوائس کا انتخاب

MPS کو یونیفائیڈ میموری میں فٹ ہونے کے لیے پوری ماڈل کی ضرورت ہوتی ہے، اس لیے `device_map="auto"` پرتوں کو CUDA جیسے CPU پر آف لوڈ نہیں کر سکتا۔ اس صورت میں، ایک چھوٹا ماڈل استعمال کرنے کی کوشش کریں۔

MPS پر وزن لوڈ کرنا تیز تر ہے اور سیفٹینسرز `0.8.0` اور PyTorch 2.9 یا بعد کے ساتھ کم میموری استعمال کرتا ہے۔ جب `device_map="mps"` یا `"auto"`، وزن کو میٹل بفرز میں بغیر کسی انٹرمیڈیٹ کاپی کے میپ کیا جاتا ہے، جو لوڈنگ کے دوران میموری فوٹ پرنٹ کو تقریباً آدھا کر دیتا ہے اور اسے تقریباً 5-6x تیز بنا دیتا ہے۔ پرانے PyTorch ورژنز پر، لوڈنگ معیاری لوڈ پاتھ پر واپس آجائے گی۔

[`Trainer`] `torch.backends.mps.is_available` کے ساتھ MPS کا خود بخود پتہ لگاتا ہے اور بغیر کسی ترتیب میں تبدیلی کے آلے کو `mps` پر سیٹ کرتا ہے۔

## مخلوط صحت سے متعلق

MPS دونوں bf16 اور fp16 مخلوط درستگی کو سپورٹ کرتا ہے (bf16 کے لیے macOS 14.0 یا اس کے بعد کی ضرورت ہوتی ہے)۔

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,  # requires macOS 14.0+
)
```



## گراف کیشے

MPS ہر منفرد ٹینسر کی شکل کے لیے ایک علیحدہ دھاتی دانا مرتب کرتا ہے اور انہیں بغیر کسی بے دخلی کی پالیسی کے گراف کیشے میں محفوظ کرتا ہے۔ متغیر لمبائی کے ان پٹس (پیڈڈ سیکوینسز، ڈائنامک بیچز) کے ساتھ ٹریننگ ہر نئی شکل پر کیشے کو بڑھاتی ہے اور آخر کار متحد میموری کو ختم کر سکتی ہے۔

اس نمو کو پابند کرنے کے لیے `torch_empty_cache_steps` کو [`TrainingArguments`] میں سیٹ کریں۔ MPS پر، [`Trainer`] ہر `torch_empty_cache_steps` قدموں پر آلے کے کیشے کے ساتھ گراف کیش کو ایک تھرو پٹ لاگت پر صاف کرتا ہے۔ آپ کو دونوں کیشز کو صاف کرنے کے لیے آپٹ ان کرنے کی ضرورت ہے۔ جب `torch_empty_cache_steps` سیٹ نہیں ہوتا ہے (پہلے سے طے شدہ)، نہ تو کیش صاف ہوتا ہے اور نہ ہی رویہ تبدیل ہوتا ہے۔

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    torch_empty_cache_steps=1,  # clear caches every step
)
```

گراف کیش کلیئرنگ کے لیے PyTorch 2.13 یا بعد کی ضرورت ہے۔ پرانے ورژنز پر، [`Trainer`] گراف کیش کال کو چھوڑتا ہے اور صرف ڈیوائس کیش کو صاف کرتا ہے۔

## اگلے اقدامات

- MPS بیک اینڈ پر پس منظر کے لیے [Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/) بلاگ پوسٹ پڑھیں۔
