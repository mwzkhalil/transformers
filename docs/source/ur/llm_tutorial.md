<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# متن کی جنریشن

[[open-in-colab]]

بڑی زبان ماڈلز (LLMs) کے لیے ٹیکسٹ جنریشن سب سے زیادہ مقبول ایپلی کیشن ہے۔ ایک LLM کو کچھ ابتدائی متن (پرامپٹ) کے ساتھ اگلا لفظ (ٹوکن) تیار کرنے کی تربیت دی جاتی ہے جس کے ساتھ اس کے اپنے تیار کردہ آؤٹ پٹ پہلے سے طے شدہ لمبائی تک یا جب یہ ترتیب کے اختتام (`EOS`) ٹوکن تک پہنچ جاتا ہے۔

Transformers میں، [`~GenerationMixin.generate`] API ٹیکسٹ جنریشن کو ہینڈل کرتا ہے، اور یہ تخلیقی صلاحیتوں کے ساتھ تمام ماڈلز کے لیے دستیاب ہے۔ یہ گائیڈ آپ کو [`~GenerationMixin.generate`] کے ساتھ ٹیکسٹ جنریشن کی بنیادی باتیں اور کچھ عام خرابیوں سے بچنے کے لیے دکھائے گا۔

> [!TIP] > درج ذیل کمانڈز کے لیے، براہ کرم یقینی بنائیں کہ [`transformers serve` چل رہا ہے](https://huggingface.co/docs/transformers/main/en/serve-cli/serving)۔ >
> ```shell
> transformers chat Qwen/Qwen2.5-0.5B-Instruct
> ```

## پہلے سے طے شدہ تخلیق

شروع کرنے سے پہلے، ان کی یادداشت کے استعمال کو کم کرنے کے لیے [bitsandbytes](https://hf.co/docs/bitsandbytes/index) واقعی بڑے ماڈلز کو کوانٹائز کرنے کے لیے انسٹال کرنا مددگار ہے۔

```bash
!pip install -U transformers bitsandbytes
```

Bitsandbytes CUDA پر مبنی GPUs کے علاوہ متعدد بیک اینڈس کو سپورٹ کرتا ہے۔ مزید جاننے کے لیے ملٹی بیک اینڈ انسٹالیشن [گائیڈ](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend) سے رجوع کریں۔

[`~PreTrainedModel.from_pretrained`] کے ساتھ LLM لوڈ کریں اور میموری کی ضروریات کو کم کرنے کے لیے درج ذیل دو پیرامیٹرز شامل کریں۔

- `device_map="auto"`Accelerates' [بڑاماڈل انفرنس](https://huggingface.co/docs/transformers/main/en/models#big-model-inference)خود کار طریقے سے شروع کرنے کے لئے خصوصیتماڈلکنکال اور لوڈنگ اور ڈسپیچنگماڈلتیز ترین ڈیوائس (GPU) سے شروع ہونے والے تمام دستیاب آلات پر وزن۔
- `quantization_config` ایک کنفیگریشن آبجیکٹ ہے جو کوانٹائزیشن سیٹنگز کی وضاحت کرتا ہے۔ یہ مثالیں bitsandbytes کو کوانٹائزیشن بیک اینڈ کے طور پر استعمال کرتی ہیں (مزید دستیاب بیک اینڈز کے لیے [Quantization](https://huggingface.co/docs/transformers/main/en/quantization/overview) سیکشن دیکھیں) اور یہ ماڈل کو [4-bits](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) میں لوڈ کرتی ہے۔

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", quantization_config=quantization_config)
```

اپنے ان پٹ کو ٹوکنائز کریں، اور [`~PreTrainedTokenizer.padding_side`] پیرامیٹر کو `"left"` پر سیٹ کریں کیونکہ LLM کو پیڈنگ ٹوکنز سے نسل جاری رکھنے کے لیے تربیت نہیں دی جاتی ہے۔ ٹوکنائزر ان پٹ ids اور توجہ کا ماسک واپس کرتا ہے۔

> [!TIP] > تاروں کی فہرست ٹوکنائزر کو دے کر ایک وقت میں ایک سے زیادہ پرامپٹ پر کارروائی کریں۔ تاخیر اور میموری میں تھوڑی قیمت پر تھرو پٹ کو بہتر بنانے کے لیے ان پٹس کو بیچ دیں۔

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
```

ٹوکن بنانے کے لیے ان پٹس کو [`~GenerationMixin.generate`] پر منتقل کریں، اور [`~PreTrainedTokenizer.batch_decode`] جنریٹ کیے گئے ٹوکنز کو ٹیکسٹ پر واپس بھیجیں۔

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
"A list of colors: red, blue, green, yellow, orange, purple, pink,"
```

## جنریشن کنفیگریشن

تمام جنریشن سیٹنگز [`GenerationConfig`] میں موجود ہیں۔ اوپر دی گئی مثال میں، جنریشن سیٹنگز [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) کی `generation_config.json` فائل سے اخذ کی گئی ہیں۔ ایک طے شدہ ضابطہ کشائی کی حکمت عملی استعمال کی جاتی ہے جب ماڈل کے ساتھ کوئی ترتیب محفوظ نہیں کی جاتی ہے۔

`generation_config` وصف کے ذریعے کنفیگریشن کا معائنہ کریں۔ یہ صرف وہی اقدار دکھاتا ہے جو پہلے سے طے شدہ ترتیب سے مختلف ہیں، اس صورت میں، `bos_token_id` اور `eos_token_id`۔

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto")
model.generation_config
GenerationConfig {
  "bos_token_id": 1,
  "eos_token_id": 2
}
```

آپ [`GenerationConfig`] میں پیرامیٹرز اور اقدار کو اوور رائیڈ کر کے [`~GenerationMixin.generate`] کو حسب ضرورت بنا سکتے ہیں۔ عام طور پر ایڈجسٹ شدہ پیرامیٹرز کے لیے [نیچے یہ سیکشن](#common-options) دیکھیں۔

```py
# enable beam search sampling strategy
model.generate(**inputs, num_beams=4, do_sample=True)
```

[`~GenerationMixin.generate`] کو بیرونی لائبریریوں یا حسب ضرورت کوڈ کے ساتھ بھی بڑھایا جا سکتا ہے:

1. `logits_processor` پیرامیٹر اگلی ٹوکن امکانی تقسیم میں ہیرا پھیری کے لیے حسب ضرورت [`LogitsProcessor`] مثالوں کو قبول کرتا ہے۔
2. `stopping_criteria` پیرامیٹرز متن کی تخلیق کو روکنے کے لیے حسب ضرورت [`StoppingCriteria`] کو سپورٹ کرتے ہیں۔
3. دیگر حسب ضرورت جنریشن طریقوں کو `custom_generate` پرچم ([docs](./generation_strategies#custom-generation-methods)) کے ذریعے لوڈ کیا جا سکتا ہے۔

تلاش، نمونے لینے، اور ضابطہ کشائی کی حکمت عملیوں کے بارے میں مزید جاننے کے لیے [جنریشن strategies](./generation_strategies) گائیڈ سے رجوع کریں۔

### محفوظ کرنا

[`GenerationConfig`] کی ایک مثال بنائیں اور ڈی کوڈنگ پیرامیٹرز کی وضاحت کریں جو آپ چاہتے ہیں۔

```py
from transformers import AutoModelForCausalLM, GenerationConfig

model = AutoModelForCausalLM.from_pretrained("my_account/my_model")
generation_config = GenerationConfig(
    max_new_tokens=50, do_sample=True, top_k=50, eos_token_id=model.config.eos_token_id
)
```

ایک مخصوص جنریشن کنفیگریشن کو محفوظ کرنے کے لیے [`~GenerationConfig.save_pretrained`] استعمال کریں اور اسے حب پر اپ لوڈ کرنے کے لیے `push_to_hub` پیرامیٹر کو `True` پر سیٹ کریں۔

```py
generation_config.save_pretrained("my_account/my_model", push_to_hub=True)
```

`config_file_name` پیرامیٹر کو خالی چھوڑ دیں۔ ایک ڈائرکٹری میں متعدد جنریشن کنفیگریشنز کو اسٹور کرتے وقت یہ پیرامیٹر استعمال کیا جانا چاہیے۔ یہ آپ کو یہ بتانے کا ایک طریقہ فراہم کرتا ہے کہ کون سی جنریشن کنفیگریشن کو لوڈ کرنا ہے۔ آپ ایک ہی ماڈل کے ساتھ استعمال کرنے کے لیے مختلف تخلیقی کاموں کے لیے مختلف کنفیگریشنز بنا سکتے ہیں (سیمپلنگ کے ساتھ تخلیقی ٹیکسٹ جنریشن، بیم سرچ کے ساتھ خلاصہ)۔

```py
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

translation_generation_config = GenerationConfig(
    num_beams=4,
    early_stopping=True,
    decoder_start_token_id=0,
    eos_token_id=model.config.eos_token_id,
    pad_token=model.config.pad_token_id,
)

translation_generation_config.save_pretrained("/tmp", config_file_name="translation_generation_config.json", push_to_hub=True)

generation_config = GenerationConfig.from_pretrained("/tmp", config_file_name="translation_generation_config.json")
inputs = tokenizer("translate English to French: Configuration files are easy to use!", return_tensors="pt")
outputs = model.generate(**inputs, generation_config=generation_config)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

<a id='common-options'></a>
## عام اختیارات

[`~GenerationMixin.generate`] ایک طاقتور ٹول ہے جسے بہت زیادہ حسب ضرورت بنایا جا سکتا ہے۔ یہ نئے صارفین کے لیے مشکل ہو سکتا ہے۔ اس سیکشن میں جنریشن کے مقبول آپشنز کی فہرست ہے جسے آپ Transformers: [`~GenerationMixin.generate`]، [`GenerationConfig`]، `pipelines`، `chat` CLI، ... میں زیادہ تر ٹیکسٹ جنریشن ٹولز میں بیان کر سکتے ہیں۔

|آپشن کا نام|قسم|آسان وضاحت|
|---|---|---|
|`max_new_tokens`|`int`|زیادہ سے زیادہ نسل کی لمبائی کو کنٹرول کرتا ہے۔ اس کی وضاحت کرنا یقینی بنائیں، کیونکہ یہ عام طور پر ایک چھوٹی قدر سے طے شدہ ہوتا ہے۔|
|`do_sample`|`bool`|اس بات کی وضاحت کرتا ہے کہ آیا نسل اگلے ٹوکن (`True`) کا نمونہ لے گی، یا اس کے بجائے لالچی ہے (`False`)۔ زیادہ تر استعمال کے معاملات میں اس پرچم کو `True` پر سیٹ کرنا چاہیے۔ مزید معلومات کے لیے [یہ گائیڈ](./generation_strategies) چیک کریں۔|
|`temperature`|`float`|اگلا منتخب ٹوکن کتنا غیر متوقع ہوگا۔ اعلی اقدار (`>0.8`) تخلیقی کاموں کے لیے اچھی ہیں، کم قدریں (جیسے `<0.4`) ان کاموں کے لیے جن کے لیے "سوچ" کی ضرورت ہوتی ہے۔ `do_sample=True` درکار ہے۔|
|`num_beams`|`int`|`>1` پر سیٹ ہونے پر، بیم سرچ الگورتھم کو فعال کرتا ہے۔ بیم کی تلاش ان پٹ گراؤنڈ کاموں پر اچھی ہے۔ مزید معلومات کے لیے [یہ گائیڈ](./generation_strategies) چیک کریں۔|
|`repetition_penalty`|`float`|اگر آپ ماڈل خود کو اکثر دہراتے ہوئے دیکھ رہے ہیں تو اسے `>1.0` پر سیٹ کریں۔ بڑی قدروں پر بڑا جرمانہ لاگو ہوتا ہے۔|
|`eos_token_id`|`list[int]`|وہ ٹوکن جو نسل کو روکنے کا سبب بنیں گے۔ پہلے سے طے شدہ قدر عام طور پر اچھی ہوتی ہے، لیکن آپ ایک مختلف ٹوکن کی وضاحت کر سکتے ہیں۔|

## نقصانات

ذیل کے حصے میں کچھ عام مسائل کا احاطہ کیا گیا ہے جن کا آپ کو ٹیکسٹ جنریشن کے دوران سامنا ہو سکتا ہے اور انہیں کیسے حل کیا جائے۔

### آؤٹ پٹ کی لمبائی

[`~GenerationMixin.generate`] بذریعہ ڈیفالٹ 20 ٹوکن واپس کرتا ہے جب تک کہ ماڈلز [`GenerationConfig`] میں وضاحت نہ کی گئی ہو۔ آؤٹ پٹ کی لمبائی کو کنٹرول کرنے کے لیے [`max_new_tokens`] پیرامیٹر کے ساتھ تیار کردہ ٹوکنز کی تعداد کو دستی طور پر سیٹ کرنے کی انتہائی سفارش کی جاتی ہے۔ [صرف ڈیکوڈر](https://hf.co/learn/nlp-course/chapter1/6?fw=pt) ماڈلز تیار کردہ ٹوکنز کے ساتھ ابتدائی پرامپٹ واپس کرتا ہے۔

```py
model_inputs = tokenizer(["A sequence of numbers: 1, 2"], return_tensors="pt").to(model.device)
```

<hfoptions id="output-length">
<hfoption id="default length">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5'
```

</hfoption>
<hfoption id="max_new_tokens">

```py
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'A sequence of numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,'
```

</hfoption>
</hfoptions>

### ضابطہ کشائی کی حکمت عملی

[`~GenerationMixin.generate`] میں پہلے سے طے شدہ ضابطہ کشائی کی حکمت عملی *لالچی تلاش* ہے، جو اگلے ممکنہ ٹوکن کا انتخاب کرتی ہے، الا یہ کہ ماڈلز [`GenerationConfig`] میں وضاحت کی گئی ہو۔ اگرچہ یہ ضابطہ کشائی کی حکمت عملی ان پٹ گراؤنڈ کاموں (ٹرانسکرپشن، ترجمہ) کے لیے اچھی طرح کام کرتی ہے، لیکن یہ زیادہ تخلیقی استعمال کے معاملات (کہانی لکھنے، چیٹ ایپلی کیشنز) کے لیے موزوں نہیں ہے۔

مثال کے طور پر، مزید متنوع نتائج پیدا کرنے کے لیے ایک [کثیراتی نمونہ سازی](./generation_strategies#sampling) حکمت عملی کو فعال کریں۔ مزید ضابطہ کشائی کی حکمت عملیوں کے لیے [جنریشن اسٹریٹیجی](./generation_strategies) گائیڈ سے رجوع کریں۔

```py
model_inputs = tokenizer(["I am a cat."], return_tensors="pt").to(model.device)
```

<hfoptions id="decoding">
<hfoption id="greedy search">

```py
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
<hfoption id="multinomial sampling">

```py
generated_ids = model.generate(**model_inputs, do_sample=True)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

</hfoption>
</hfoptions>

### پیڈنگ سائیڈ

اگر ان کی لمبائی ایک جیسی نہیں ہے تو ان پٹ کو پیڈ کرنے کی ضرورت ہے۔ لیکن LLMs کو پیڈنگ ٹوکنز سے جنریشن جاری رکھنے کے لیے تربیت نہیں دی جاتی ہے، جس کا مطلب ہے کہ [`~PreTrainedTokenizer.padding_side`] پیرامیٹر کو ان پٹ کے بائیں جانب سیٹ کرنے کی ضرورت ہے۔ سٹیٹ اسپیس اور ہائبرڈ ماڈلز کے لیے دائیں پیڈنگ بدتر ہے کیونکہ ان کی بار بار آنے والی حالت ٹریلنگ پیڈ ٹوکنز پر اپ ڈیٹ ہوتی رہتی ہے، اس لیے کیش اب پرامپٹ کے اختتام سے میل نہیں کھاتا۔

<hfoptions id="padding">
<hfoption id="right pad">

```py
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 33333333333'
```

</hfoption>
<hfoption id="left pad">

```py
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model_inputs = tokenizer(
    ["1, 2, 3", "A, B, C, D, E"], padding=True, return_tensors="pt"
).to(model.device)
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
'1, 2, 3, 4, 5, 6,'
```

</hfoption>
</hfoptions>

### فوری فارمیٹ

کچھ ماڈلز اور ٹاسک ایک مخصوص ان پٹ پرامپٹ فارمیٹ کی توقع کرتے ہیں، اور اگر فارمیٹ غلط ہے تو، ماڈل ایک سب سے بہترین آؤٹ پٹ لوٹاتا ہے۔ آپ [prompt engineering](./tasks/prompting) گائیڈ میں اشارہ کرنے کے بارے میں مزید جان سکتے ہیں۔

مثال کے طور پر، ایک چیٹ ماڈل ان پٹ کو بطور [چیٹ ٹیمپلیٹ](./chat_templating) کی توقع رکھتا ہے۔ آپ کے پرامپٹ میں `role` اور `content` شامل ہونا چاہیے تاکہ اس بات کی نشاندہی کی جا سکے کہ گفتگو میں کون حصہ لے رہا ہے۔ اگر آپ اپنے پرامپٹ کو ایک سٹرنگ کے طور پر پاس کرنے کی کوشش کرتے ہیں، تو ماڈل ہمیشہ متوقع آؤٹ پٹ واپس نہیں کرتا ہے۔

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-alpha")
model = AutoModelForCausalLM.from_pretrained(
    "HuggingFaceH4/zephyr-7b-alpha", device_map="auto", quantization_config=BitsAndBytesConfig(load_in_4bit=True)
)
```

<hfoptions id="format">
<hfoption id="no format">

```py
prompt = """How many cats does it take to change a light bulb? Reply as a pirate."""
model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
input_length = model_inputs.input_ids.shape[1]
generated_ids = model.generate(**model_inputs, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Aye, matey! 'Tis a simple task for a cat with a keen eye and nimble paws. First, the cat will climb up the ladder, carefully avoiding the rickety rungs. Then, with"
```

</hfoption>
<hfoption id="chat template">

```py
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a pirate",
    },
    {"role": "user", "content": "How many cats does it take to change a light bulb?"},
]
model_inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
input_length = model_inputs["input_ids"].shape[1]
generated_ids = model.generate(**model_inputs, do_sample=True, max_new_tokens=50)
print(tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)[0])
"Arr, matey! According to me beliefs, 'twas always one cat to hold the ladder and another to climb up it an’ change the light bulb, but if yer looking to save some catnip, maybe yer can
```

</hfoption>
</hfoptions>

## وسائل

کچھ مزید مخصوص اور خصوصی ٹیکسٹ جنریشن لائبریریوں کے لیے ذیل میں ایک نظر ڈالیں۔

- [Optimum](https://github.com/huggingface/optimum): Transformers کی توسیع تربیت کو بہتر بنانے پر مرکوز ہے اور انفرنس مخصوص ہارڈ ویئر آلات پر
- [Outlines](https://github.com/dottxt-ai/outlines): محدود ٹیکسٹ جنریشن کے لیے ایک لائبریری (مثال کے طور پر JSON فائلیں بنائیں)۔
- [SynCode](https://github.com/uiuc-focal-lab/syncode): سیاق و سباق سے پاک گرامر گائیڈڈ جنریشن (JSON, SQL, Python) کے لیے ایک لائبریری۔
- ٹیکسٹ جنریشن انفرنس](https://github.com/huggingface/text-generation-inference): LLMs کے لیے پروڈکشن کے لیے تیار سرور۔
- [ٹیکسٹ جنریشن ویب UI](https://github.com/oobabooga/text-generation-webui): ٹیکسٹ جنریشن کے لیے ایک Gradio ویب UI۔
- [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo): ٹیکسٹ جنریشن کو کنٹرول کرنے کے لیے اضافی لاگٹس پروسیسرز۔
