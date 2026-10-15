<!--Copyright 2026 The HuggingFace Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance 
with
the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is 
distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for 
the
specific language governing permissions and limitations under the License.
⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that 
may not be
rendered properly in your Markdown viewer.
-->

# ٹوکنائزرز

A ٹوکنائزر متن کو ٹینسر میں تبدیل کرتا ہے، جو کہ ماڈل کے ان پٹ ہوتے ہیں۔ یہ متن کو معمول پر لاتا ہے اور تقسیم کرتا ہے، ٹوکنائزیشن الگورتھم کا اطلاق کرتا ہے، خصوصی ٹوکنز شامل کرتا ہے، اور آؤٹ پٹ آئی ڈی کو واپس متن میں ڈی کوڈ کرتا ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

اس گائیڈ میں لوڈنگ، انکوڈنگ، ڈی کوڈنگ، بیچ پروسیسنگ، اور دستیاب ٹوکنائزر بیک اینڈز شامل ہیں۔

## ایک ٹوکنائزر لوڈ کریں

ٹوکنائزر کو [`AutoTokenizer`] کلاس یا ماڈل مخصوص ٹوکنائزر کلاس کے ساتھ لوڈ کریں۔

<hfoptions id="tokenizers">
<hfoption id="AutoTokenizer">

[`AutoTokenizer.from_pretrained`] ماڈل ترتیب کو پڑھتا ہے، درست ٹوکنائزر کلاس کو حل کرتا ہے، اور اس کی ایک مثال واپس کرتا ہے۔ آپ کو ٹوکنائزر کلاس کو پہلے سے جاننے کی ضرورت نہیں ہے۔ زیادہ تر ٹوکنائزرز [`TokenizersBackend`] کے ذیلی طبقے کو حل کرتے ہیں، [ٹوکنائزرز](https://huggingface.co/docs/tokenizers/index) لائبریری سے ایک تیز زنگ پر مبنی ٹوکنائزر۔

[`AutoTokenizer`] کے ساتھ لوڈ کرنا تجویز کردہ طریقہ ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
```

</hfoption>
<hfoption id="model-specific tokenizer">

اےماڈلمخصوص ٹوکنائزیشن کلاس پہلے سے تشکیل شدہ ہے۔[`TokenizersBackend`]جو درست ٹوکنائزیشن کنفیگریشن کا استعمال کرتا ہے (نارملائزر، پری-ٹوکنائزر, خصوصی ٹوکن کنونشنز وغیرہ) aماڈلکے ساتھ تربیت دی گئی تھی۔

تربیت کے لیے خالی ٹوکنائزر شروع کرنے یا ماڈل مخصوص دلائل جیسے `vocab` یا `merges` پاس کرنے کے لیے ماڈل مخصوص کلاس استعمال کریں (دیکھیں [اپنی مرضی کے مطابق بنانا طریقہ جاننے کے لیے ٹوکنائزرز](https://huggingface.co/docs/transformers/main/en/custom_tokenizers) گائیڈ)۔ ایک خالی ٹوکنائزر کم سے کم ہے اور اس میں صرف ماڈل کے خصوصی ٹوکنز جیسے `<pad>`، `<eos>`، یا `<bos>` ہوتے ہیں۔

```py
from transformers import GemmaTokenizer

tokenizer = GemmaTokenizer()
corpus = [
    ["Sphinx of black quartz, judge my vow."],
    ["Pack my box with five dozen liquor jugs."],
    ["How vexingly quick daft zebras jump!"],
]
new_tokenizer = tokenizer.train_new_from_iterator(corpus, vocab_size=1000)
```

</hfoption>
</hfoptions>

## انکوڈ اور ڈی کوڈ کریں۔

[`TokenizersBackend.__call__`] طریقہ متن یا متن کے بیچ کو `input_ids`، `attention_mask`، اور دیگر ماڈل ان پٹ میں انکوڈ کرتا ہے۔ یہ پیڈنگ، ٹرنکیشن، اور خصوصی ٹوکن اندراج کو بھی کنٹرول کرتا ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer("Sphinx of black quartz, judge my vow.", return_tensors="pt")
{
    'input_ids': tensor([[     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265]]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

[`TokenizersBackend.encode`] ملتا جلتا ہے لیکن صرف `input_ids` لوٹاتا ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
input_ids = tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
```

[`TokenizersBackend.decode`] ایک واحد ترتیب یا ٹوکنائزڈ `input_ids` کے بیچ کو واپس متن میں تبدیل کرتا ہے۔

```py
tokenizer.decode(input_ids)
'<bos>Sphinx of black quartz, judge my vow.'
```

[`TokenizersBackend.decode`] ٹوکنائزیشن کے عین وقفہ کو محفوظ رکھتا ہے۔ اوقاف سے پہلے خالی جگہوں کو ہٹانے کے لیے `clean_up_tokenization_spaces` اور آؤٹ پٹ سے خصوصی ٹوکن ہٹانے کے لیے `skip_special_tokens` سیٹ کریں۔

```py
tokenizer.decode(input_ids, skip_special_tokens=True)
'Sphinx of black quartz, judge my vow.'
```

## خصوصی ٹوکن

خصوصی ٹوکن ایک ترتیب میں ساختی حدود کو نشان زد کرتے ہیں، جیسے ترتیب کے آغاز یا پیڈنگ پوزیشنز۔ ہر ماڈل مخصوص ٹوکنز کے اپنے سیٹ کی وضاحت کرتا ہے۔ جب آپ اسے کال کرتے ہیں تو ٹوکنائزر انہیں شامل کرتا ہے۔

```py
input_ids = tokenizer.encode("Sphinx of black quartz, judge my vow.")
[2, 235277, 82913, 576, 2656, 30407, 235269, 11490, 970, 29871, 235265]
tokenizer.decode(input_ids)
'<bos>Sphinx of black quartz, judge my vow.'
```

`extra_special_tokens` دلیل کے ساتھ اضافی نامزد خصوصی ٹوکنز رجسٹر کریں۔ ملٹی موڈل ماڈلز انہیں تصاویر، ویڈیو یا آڈیو کے لیے پلیس ہولڈرز کے طور پر استعمال کریں۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-pt",
    extra_special_tokens={"image_token": "<image>"}
)
```

## بیچ پروسیسنگ

بیچ پروسیسنگ ایک ہی کال میں متعدد سلسلے کو ٹوکنائز کرتی ہے۔ [`TokenizersBackend`] بڑے بیچوں کو تیزی سے ہینڈل کرتا ہے کیونکہ اس کا زنگ پر مبنی بیک اینڈ تمام تھریڈز کے کام کو متوازی کرتا ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt"
)
```

بیچ پروسیسنگ کے لیے ایک ہی لمبائی کا اشتراک کرنے کے لیے تمام ترتیب کی ضرورت ہوتی ہے۔ پیڈنگ اور ٹرنکیشن مختلف لمبائی کے سلسلے کو سنبھالنے کی حکمت عملی ہیں۔

### پیڈنگ

پیڈنگ خصوصی ٹوکنز کو جوڑتی ہے اس لیے چھوٹے سلسلے بیچ میں سب سے طویل ترتیب سے ملتے ہیں۔ توجہ کا ماسک پیڈنگ پوزیشنز کو `0` کے طور پر نشان زد کرتا ہے لہذا ماڈل انہیں نظر انداز کر دیتا ہے۔ `padding=True` کو سب سے طویل ترتیب پر پیڈ پر سیٹ کریں یا `max_length` کو ایک مقررہ سائز میں پیڈ پر منتقل کریں۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656,  30407, 235269,  11490,    970,  29871, 235265],
        [     0,      2,   6519,    970,   3741,    675,   4105,  25955,  42184, 225789, 235265],
        [     0,      2,   2299,  73378,  17844,   4320, 224463,   4949,  48977,  9902, 235341]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
}
```

> [!نوٹ] > بڑی زبان ماڈلز پیڈ *بائیں* سائیڈ پر ہے تاکہ جنریشن میں خلل ڈالنے سے بچ سکے، جو *دائیں* سائیڈ سے اگلے ٹوکن کی پیشین گوئی کرتا ہے۔

### تراشنا

ٹرنکیشن کلپس ٹوکنز اس لیے ایک ترتیب زیادہ سے زیادہ لمبائی میں فٹ ہو جاتی ہے۔ `truncation=True` سیٹ کریں اور اسے فعال کرنے کے لیے `max_length` کی وضاحت کریں۔

پیڈنگ اور ٹرنکیشن ایک ساتھ کام کرتے ہیں۔ مختصر سیکوینس پیڈنگ ٹوکن حاصل کرتے ہیں جبکہ لمبے سیکوینس پچھلے ٹوکن کھو دیتے ہیں۔ ایک ساتھ مل کر، وہ ایک پیک شدہ مستطیل ٹینسر تیار کرتے ہیں۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer(
    [
        "Sphinx of black quartz, judge my vow.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!"
    ],
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=5
)
{
    'input_ids': tensor([
        [     2, 235277,  82913,    576,   2656],
        [     2,   6519,    970,   3741,    675],
        [     2,   2299,  73378,  17844,   4320]
    ]),
    'attention_mask': tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])
}
```

## بیک اینڈز

ہر ماڈل ٹوکنائزر ایک فائل میں بیان کیا گیا ہے اور چار ٹوکنائزیشن بیک اینڈس کو سپورٹ کرتا ہے۔


|پسدید|نفاذ|تفصیل|
| ------------------------ | ------------------------------------------------------------- | ---------------------------------------------- |
|[`TokenizersBackend`]|[ٹوکنائزرز](https://huggingface.co/docs/tokenizers)|زیادہ تر ماڈلز کے لیے ڈیفالٹ|
|[`SentencePieceBackend`]|[SentencePiece](https://github.com/google/sentencepiece)|ماڈلز SentencePiece کی ضرورت ہے۔|
|[`PythonBackend`]|ازگر|ماڈلز خصوصی حسب ضرورت ٹوکنائزرز|
|[`MistralCommonBackend`]|mistral-common](https://mistralai.github.io/mistral-common/)|Mistral اور Pixtral ماڈلز|

تمام بیک اینڈز [`PreTrainedTokenizerBase`] سے وراثت میں ملتے ہیں اور انکوڈنگ، ڈی کوڈنگ، پیڈنگ، تراشنے، محفوظ کرنے اور لوڈ کرنے کے لیے ایک جیسے APIs کا اشتراک کرتے ہیں۔ فرق یہ ہے کہ کون سی ٹوکنائزیشن پائپ لائن نیچے چلتی ہے۔

[`AutoTokenizer`] بہترین دستیاب بیک اینڈ کا انتخاب کرتا ہے جب آپ [`~AutoTokenizer.from_pretrained`] کال کرتے ہیں۔

1. یہ `tokenizer_class` فیلڈ کے لیے `tokenizer_config.json` فائل کو پڑھتا ہے۔
2. رجسٹری کلاس کے نام سے `tokenizer_class` مماثل ہے۔ حل شدہ کلاس چار بیک اینڈز میں سے ایک سے وراثت میں ملتی ہے۔ مثال کے طور پر، [`GemmaTokenizer`] [`TokenizersBackend`] سے وراثت میں ملتا ہے، اور [`SiglipTokenizer`] [`SentencePieceBackend`] سے وراثت میں ملتا ہے۔
کچھ ماڈلز، GLM کی طرح، براہ راست [`TokenizersBackend`] پر نقشہ بناتے ہیں کیونکہ `tokenizer.json` فائل پائپ لائن کو مکمل طور پر بیان کرتی ہے۔ [`GemmaTokenizer`] ایک ذیلی کلاس کے طور پر موجود ہے کیونکہ یہ Python میں اضافی ماڈل مخصوص ترتیبات کی وضاحت کرتا ہے جو `tokenizer.json` کیپچر نہیں کرتا ہے۔
    When a backend like mistral-common isn't installed, [`AutoTokenizer`] falls back to [`TokenizersBackend`].

### TokenizersBackend پر فال بیک

کچھ ماڈلز کوئی وقف شدہ ٹوکنائزر کلاس نہیں ہے، اور کچھ چیک پوائنٹس میں `tokenizer_class` ہے جو ان کی اصل `tokenizer.json` فائل سے مماثل نہیں ہے۔ [`AutoTokenizer`] پائپ لائن کو `tokenizer.json` سے عام [`TokenizersBackend`] کے ذریعے لوڈ کر کے دونوں صورتوں کو ہینڈل کرتا ہے۔ یہ حب سے کلاس کے نام پر سیریلائزڈ ٹوکنائزر کو ترجیح دیتا ہے، جو ان مسائل کو ٹھیک کرتا ہے جیسے کہ جب کوئی مماثل `tokenizer_class` غلط ٹوکن آئی ڈی تیار کرتا ہے۔

A چیک پوائنٹ تین وجوہات میں سے ایک کی وجہ سے ایک عام `TokenizersBackend` میں حل کرتا ہے۔


|وجہ|رویہ|مثالیں|
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
|کوئی وقف شدہ ٹوکنائزر کلاس نہیں ہے۔|ماڈل قسم کے نقشے سیدھے `TokenizersBackend` پر آتے ہیں کیونکہ `tokenizer.json` پائپ لائن کی مکمل وضاحت کرتا ہے۔|GLM, Granite, OLMo 2, GPT BigCode|
|معلوم غلط حب ٹوکنائزر کلاس|حب پر ریکارڈ کردہ `tokenizer_class` ماڈل قسم کے لیے غلط ہے، لہذا `AutoTokenizer` اسے نظر انداز کر دیتا ہے اور اس کی بجائے `tokenizer.json` لوڈ کرتا ہے۔|ڈیپ سیک V3، LLaVA، Qwen2، ModernBERT|
|مخصوص چیک پوائنٹ اوور رائیڈ|ایک چیک پوائنٹ جس کی حب کنفیگریشن کو ابھی بھی درست کرنے کی ضرورت ہے اس کی ماڈل آئی ڈی سے مماثل ہے اور بیک اینڈ پر مجبور کیا جاتا ہے۔|`deepseek-ai/DeepSeek-R1-Distill-*`, `Salesforce/blip2-*`, `google/umt5-small`|

متاثرہ ماڈل اقسام اور چیک پوائنٹس بڑھتے جاتے ہیں جیسے ہی حب پر کنفیگرز درست کیے جاتے ہیں۔ موجودہ سیٹ کے لیے، `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` اور `MODEL_IDS_TO_TOKENIZERS_BACKEND` تعریفیں [tokenization_auto.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/auto/tokenization_auto.py) میں دیکھیں۔

فال بیک خودکار ہے اور آپ کے [`~AutoTokenizer.from_pretrained`] کال کرنے کے طریقے کو تبدیل نہیں کرتا ہے۔ نتیجے میں آنے والا ٹوکنائزر انکوڈ اور ڈی کوڈ کرتا ہے جیسا کہ `tokenizer.json` وضاحت کرتا ہے۔

چیک کریں کہ ٹوکنائزر `backend` پراپرٹی کے ساتھ کون سا بیک اینڈ استعمال کر رہا ہے۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
tokenizer.backend
'tokenizers'
```

## ٹوکنائزر فن تعمیر کا معائنہ کریں۔

ٹوکنائزر کے اندرونی اجزاء (نارملائزر، پری ٹوکنائزر، ماڈل، ڈیکوڈر) کا `_tokenizer` وصف کے ساتھ معائنہ کریں۔

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
print(tokenizer._tokenizer.normalizer)
print(tokenizer._tokenizer.pre_tokenizer)
print(tokenizer._tokenizer.model)
print(tokenizer._tokenizer.decoder)
```

## وسائل

- [Transformers v5](https://huggingface.co/blog/tokenizers) پوسٹ میں ٹوکنائزیشن نئے ٹوکنائزیشن بیک اینڈز کے پیچھے محرک پر بحث کرتی ہے۔
- ٹوکنائزیشن تبدیلیوں کا جائزہ لینے کے لیے [مائیگریشن گائیڈ](https://github.com/huggingface/transformers/blob/main/MIGRATION_GUIDE_V5.md#tokenization) کا جائزہ لیں۔
