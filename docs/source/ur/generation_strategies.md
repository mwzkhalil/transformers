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

# ڈیکوڈنگ کے طریقے

ضابطہ کشائی کی حکمت عملی بتاتی ہے کہ کس طرح ماڈل کو اگلا تیار کردہ ٹوکن منتخب کرنا چاہیے۔ ضابطہ کشائی کی حکمت عملیوں کی بہت سی قسمیں ہیں، اور مناسب کو منتخب کرنے سے تیار کردہ متن کے معیار پر نمایاں اثر پڑتا ہے۔

یہ گائیڈ آپ کو Transformers میں دستیاب مختلف ضابطہ کشائی کی حکمت عملیوں کو سمجھنے میں مدد کرے گا اور انہیں کیسے اور کب استعمال کرنا ہے۔

## بنیادی ضابطہ کشائی کے طریقے

یہ ضابطہ کشائی کے اچھے طریقے ہیں، اور ٹیکسٹ جنریشن کے کاموں کے لیے آپ کا نقطہ آغاز ہونا چاہیے۔

### لالچی تلاش

لالچی تلاش پہلے سے طے شدہ ضابطہ کشائی کی حکمت عملی ہے۔ یہ ہر قدم پر اگلا ممکنہ ٹوکن منتخب کرتا ہے۔ جب تک کہ [`GenerationConfig`] میں وضاحت نہ کی جائے، یہ حکمت عملی زیادہ سے زیادہ 20 نئے ٹوکن تیار کرتی ہے۔

لالچی تلاش نسبتاً کم نتائج والے کاموں کے لیے اچھی طرح سے کام کرتی ہے جہاں تخلیقی صلاحیتوں کو ترجیح نہیں دی جاتی ہے۔ تاہم، طویل ترتیب پیدا کرنے پر یہ ٹوٹ جاتا ہے کیونکہ یہ خود کو دہرانا شروع کر دیتا ہے۔

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to default length because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=20)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company that provides a suite of tools and services for building, deploying, and maintaining natural language processing'
```

<a id='sampling'></a>
### سیمپلنگ

سیمپلنگ، یا کثیر الجہتی نمونے، تصادفی طور پر ماڈل کی پوری ذخیرہ الفاظ پر احتمالی تقسیم کی بنیاد پر ایک ٹوکن کا انتخاب کرتا ہے (جیسا کہ لالچی تلاش میں سب سے زیادہ امکان والے ٹوکن کے برخلاف)۔ اس کا مطلب ہے کہ غیر صفر امکان کے ساتھ ہر ٹوکن کو منتخب کرنے کا موقع ملتا ہے۔ نمونے لینے کی حکمت عملی تکرار کو کم کرتی ہے اور زیادہ تخلیقی اور متنوع نتائج پیدا کر سکتی ہے۔

`do_sample=True` اور `num_beams=1` کے ساتھ کثیر الجہتی نمونے کو فعال کریں۔

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, num_beams=1)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
'Hugging Face is an open-source company 🤗\nWe are open-source and believe that open-source is the best way to build technology. Our mission is to make AI accessible to everyone, and we believe that open-source is the best way to achieve that.'
```

### بیم کی تلاش

شہتیر کی تلاش ہر ایک قدم پر کئی پیدا شدہ ترتیبوں (بیم) کا ٹریک رکھتی ہے۔ ایک خاص تعداد کے مراحل کے بعد، یہ سب سے زیادہ *مجموعی* امکان کے ساتھ ترتیب کا انتخاب کرتا ہے۔ لالچی تلاش کے برعکس، یہ حکمت عملی "آگے دیکھ" سکتی ہے اور مجموعی طور پر زیادہ امکان کے ساتھ ایک ترتیب چن سکتی ہے یہاں تک کہ اگر ابتدائی ٹوکن کا امکان کم ہو۔ یہ ان پٹ گراؤنڈ کاموں کے لیے بہترین موزوں ہے، جیسے کسی تصویر کو بیان کرنا یا اسپیچ ریکگنیشن۔ آپ بیم کی تلاش کے ساتھ `do_sample=True` کو بھی ہر قدم پر نمونے کے لیے استعمال کر سکتے ہیں، لیکن شہتیر کی تلاش اب بھی لالچ کے ساتھ مراحل کے درمیان کم امکانی ترتیب کو کاٹ دے گی۔

> [!TIP] > یہ دیکھنے کے لیے کہ [beam search visualizer](https://huggingface.co/spaces/m-ric/beam_search_visualizer) چیک کریں کہ بیم کی تلاش کیسے کام کرتی ہے۔

`num_beams` پیرامیٹر کے ساتھ بیم تلاش کو فعال کریں (1 سے زیادہ ہونا چاہیے ورنہ یہ لالچی تلاش کے مترادف ہے)۔

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator

device = Accelerator().device

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
inputs = tokenizer("Hugging Face is an open-source company", return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype=torch.float16).to(device)
# explicitly set to 100 because Llama2 generation length is 4096
outputs = model.generate(**inputs, max_new_tokens=50, num_beams=2)
tokenizer.batch_decode(outputs, skip_special_tokens=True)
"['Hugging Face is an open-source company that develops and maintains the Hugging Face platform, which is a collection of tools and libraries for building and deploying natural language processing (NLP) models. Hugging Face was founded in 2018 by Thomas Wolf']"
```

<a id='custom-generation-methods'></a>
## کسٹم جنریشن کے طریقے

کسٹم جنریشن کے طریقے خصوصی طرز عمل کو فعال کرتے ہیں جیسے:

- ماڈل کو سوچنا جاری رکھیں اگر یہ غیر یقینی ہے؛
- رول بیک جنریشن اگر ماڈل پھنس جائے؛
- اپنی مرضی کے مطابق منطق کے ساتھ خصوصی ٹوکن ہینڈل؛
- خصوصی KV کیچز کا استعمال کریں؛

ہم ایک مخصوص ماڈل ٹیگ اور فائل کی ساخت کو فرض کرتے ہوئے ماڈل ذخیروں کے ذریعے حسب ضرورت جنریشن کے طریقوں کو فعال کرتے ہیں (نیچے ذیلی سیکشن دیکھیں)۔ یہ خصوصیت [کسٹم ماڈلنگ کوڈ](https://huggingface.co/docs/transformers/main/en/models#custom-models) کی توسیع ہے اور اس طرح، `trust_remote_code=True` سیٹنگ کی ضرورت ہے۔

اگر ماڈل ذخیرہ میں حسب ضرورت جنریشن کا طریقہ ہے، تو اسے آزمانے کا سب سے آسان طریقہ یہ ہے کہ ماڈل لوڈ کریں اور اس سے تخلیق کریں:

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

# `transformers-community/custom_generate_example` holds a copy of `Qwen/Qwen2.5-0.5B-Instruct`, but
# with custom generation code -> calling `generate` uses the custom generation method!
tokenizer = AutoTokenizer.from_pretrained("transformers-community/custom_generate_example")
model = AutoModelForCausalLM.from_pretrained(
    "transformers-community/custom_generate_example", device_map="auto", trust_remote_code=True
)

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# The custom generation method is a minimal greedy decoding implementation. It also prints a custom message at run time.
gen_out = model.generate(**inputs)
# you should now see its custom message, "✨ using a custom generation method ✨"
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True))
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

ماڈل اپنی مرضی کے مطابق جنریشن کے طریقوں کے ساتھ ریپوزٹریز کی ایک خاص خاصیت ہوتی ہے: ان کے جنریشن کا طریقہ **کسی بھی** ماڈل سے [`~GenerationMixin.generate`] کی `custom_generate` دلیل سے لوڈ کیا جا سکتا ہے۔ اس کا مطلب ہے کہ کوئی بھی صارف کو اضافی Python پیکجز انسٹال کرنے کی ضرورت کے بغیر، کسی بھی Transformers ماڈل کے ساتھ ممکنہ طور پر کام کرنے کے لیے اپنا حسب ضرورت جنریشن طریقہ بنا اور شیئر کر سکتا ہے۔

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")

inputs = tokenizer(["The quick brown"], return_tensors="pt").to(model.device)
# `custom_generate` replaces the original `generate` by the custom generation method defined in
# `transformers-community/custom_generate_example`
gen_out = model.generate(**inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True)
print(tokenizer.batch_decode(gen_out, skip_special_tokens=True)[0])
'The quick brown fox jumps over a lazy dog, and the dog is a type of animal. Is'
```

آپ کو ریپوزٹری کی `README.md` فائل پڑھنی چاہیے جس میں کسٹم جنریشن کی حکمت عملی موجود ہے یہ دیکھنے کے لیے کہ اگر وہ موجود ہیں تو نئے دلائل اور آؤٹ پٹ قسم کے فرق کیا ہیں۔ بصورت دیگر، آپ فرض کر سکتے ہیں کہ یہ بیس [`~GenerationMixin.generate`] طریقہ کی طرح کام کرتا ہے۔

> [!TIP] > آپ اپنی مرضی کے مطابق جنریشن کے تمام طریقے [ان کے حسب ضرورت ٹیگ کو تلاش کرکے تلاش کر سکتے ہیں۔](https://huggingface.co/models?other=custom_generate), `custom_generate`.

مثال کے طور پر حب ریپوزٹری [Transformers-community/custom_generate_example](https://huggingface.co/transformers-community/custom_generate_example) پر غور کریں۔ `README.md` کہتا ہے کہ اس میں ایک اضافی ان پٹ دلیل ہے، `left_padding`، جو پرامپٹ سے پہلے متعدد پیڈنگ ٹوکنز کا اضافہ کرتا ہے۔

```py
gen_out = model.generate(
    **inputs, custom_generate="transformers-community/custom_generate_example", trust_remote_code=True, left_padding=5
)
print(tokenizer.batch_decode(gen_out)[0])
'<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>The quick brown fox jumps over the lazy dog.\n\nThe sentence "The quick'
```

اگر حسب ضرورت طریقہ نے Python کی ضروریات کو پن کیا ہے جو آپ کا ماحول پورا نہیں کرتا ہے، تو آپ کو غائب ضروریات کے بارے میں ایک استثنا ملے گا۔ مثال کے طور پر، [Transformers-community/custom_generate_bad_requirements](https://huggingface.co/transformers-community/custom_generate_bad_requirements) اس کی `custom_generate/requirements.txt` فائل میں بیان کردہ ضروریات کا ایک ناممکن سیٹ ہے، اور اگر آپ اسے چلانے کی کوشش کرتے ہیں تو آپ کو نیچے خرابی کا پیغام نظر آئے گا۔

```text
ImportError: Missing requirements in your local environment for `transformers-community/custom_generate_bad_requirements`:
foo (installed: None)
bar==0.0.0 (installed: None)
torch>=99.0 (installed: 2.6.0)
```

اس کے مطابق آپ کی ازگر کی ضروریات کو اپ ڈیٹ کرنے سے یہ ایرر میسج ختم ہو جائے گا۔

### اپنی مرضی کے مطابق تخلیق کا طریقہ بنانا

نئی تخلیق کا طریقہ بنانے کے لیے، آپ کو ایک نیا [**ماڈل**](https://huggingface.co/new) ذخیرہ بنانے اور اس میں چند فائلوں کو دھکیلنے کی ضرورت ہے۔

1. ماڈل جس کے ساتھ آپ نے اپنا جنریشن طریقہ ڈیزائن کیا ہے۔
2. `custom_generate/generate.py`، جس میں آپ کے حسب ضرورت جنریشن کے طریقہ کار کے لیے تمام منطق شامل ہیں۔
3. `custom_generate/requirements.txt`، اختیاری طور پر نئے Python کے تقاضے شامل کرنے اور/یا مخصوص ورژنز کو اپنے طریقہ کار کو صحیح طریقے سے استعمال کرنے کے لیے استعمال کیا جاتا ہے۔
4. `README.md`، جہاں آپ کو `custom_generate` ٹیگ شامل کرنا چاہیے اور یہاں اپنے حسب ضرورت طریقہ کے کسی بھی نئے دلائل یا آؤٹ پٹ قسم کے فرق کو دستاویز کرنا چاہیے۔

تمام مطلوبہ فائلوں کو شامل کرنے کے بعد، آپ کا ذخیرہ اس طرح نظر آنا چاہیے۔

```text
your_repo/
├── README.md          # include the 'custom_generate' tag
├── config.json
├── ...
└── custom_generate/
    ├── generate.py
    └── requirements.txt
```

#### بیس ماڈل شامل کرنا

آپ کے کسٹم جنریشن کے طریقہ کار کا نقطہ آغاز کسی دوسرے کی طرح ایک ماڈل ذخیرہ ہے۔ اس ریپوزٹری میں شامل کرنے کے لیے ماڈل وہ ماڈل ہونا چاہیے جس کے ساتھ آپ نے اپنا طریقہ ڈیزائن کیا ہے، اور اس کا مقصد کام کرنے والی خود ساختہ ماڈل-جنریٹ جوڑی کا حصہ ہونا ہے۔ جب اس ریپوزٹری میں ماڈل لوڈ ہو جائے گا، تو آپ کا کسٹم جنریشن کا طریقہ `generate` کو اوور رائیڈ کر دے گا۔ پریشان نہ ہوں -- آپ کا جنریشن طریقہ اب بھی کسی دوسرے Transformers ماڈل کے ساتھ لوڈ کیا جا سکتا ہے، جیسا کہ اوپر والے حصے میں بیان کیا گیا ہے۔

اگر آپ صرف ایک موجودہ ماڈل کاپی کرنا چاہتے ہیں، تو آپ کر سکتے ہیں۔

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("source/model_repo")
model = AutoModelForCausalLM.from_pretrained("source/model_repo")
tokenizer.save_pretrained("your/generation_method", push_to_hub=True)
model.save_pretrained("your/generation_method", push_to_hub=True)
```

#### generate.py

یہ آپ کی تخلیق کے طریقہ کار کا بنیادی حصہ ہے۔ یہ *لازمی طور پر* میں `generate` نام کا طریقہ ہونا چاہیے، اور اس طریقہ *ضروری* میں `model` دلیل اس کی پہلی دلیل کے طور پر ہونی چاہیے۔ `model` ماڈل مثال ہے، جس کا مطلب ہے کہ آپ کو ماڈل میں موجود تمام صفات اور طریقوں تک رسائی حاصل ہے، بشمول [`GenerationMixin`] میں بیان کردہ (جیسے کہ بنیاد `generate` طریقہ)۔

> [!انتباہ] > `generate.py` کو `custom_generate` نامی فولڈر میں رکھا جانا چاہیے، نہ کہ ریپوزٹری کے روٹ لیول پر۔ اس خصوصیت کے لیے فائل کے راستے ہارڈ کوڈ کیے گئے ہیں۔

ہڈ کے نیچے، جب بنیاد[`~GenerationMixin.generate`]طریقہ کو a کے ساتھ کہا جاتا ہے۔`custom_generate`argument، یہ پہلے اپنی ازگر کی ضروریات (اگر کوئی ہے) کو چیک کرتا ہے، پھر کسٹم کا پتہ لگاتا ہے۔`generate`میں طریقہ`generate.py`، اور آخر میں اپنی مرضی کو کال کرتا ہے۔`generate`. تمام موصول ہونے والے دلائل اور`model`اپنی مرضی کے مطابق بھیجے جاتے ہیں۔`generate`طریقہ، اپنی مرضی کی تخلیق کو متحرک کرنے کے لیے استعمال ہونے والے دلائل کے استثنا کے ساتھ (`trust_remote_code`اور`custom_generate`).

اس کا مطلب ہے کہ آپ کے `generate` میں اصل اور حسب ضرورت دلائل کا مرکب ہو سکتا ہے (نیز آؤٹ پٹ کی مختلف قسم) جیسا کہ ذیل میں دکھایا گیا ہے۔

```py
import torch

def generate(model, input_ids, generation_config=None, left_padding=None, **kwargs):
    generation_config = generation_config or model.generation_config  # default to the model generation config
    cur_length = input_ids.shape[1]
    max_length = generation_config.max_length or cur_length + generation_config.max_new_tokens

    # Example of custom argument: add `left_padding` (integer) pad tokens before the prompt
    if left_padding is not None:
        if not isinstance(left_padding, int) or left_padding < 0:
            raise ValueError(f"left_padding must be an integer larger than 0, but is {left_padding}")

        pad_token = kwargs.pop("pad_token", None) or generation_config.pad_token_id or model.config.pad_token_id
        if pad_token is None:
            raise ValueError("pad_token is not defined")
        batch_size = input_ids.shape[0]
        pad_tensor = torch.full(size=(batch_size, left_padding), fill_value=pad_token).to(input_ids.device)
        input_ids = torch.cat((pad_tensor, input_ids), dim=1)
        cur_length = input_ids.shape[1]

    # Simple greedy decoding loop
    while cur_length < max_length:
        logits = model(input_ids).logits
        next_token_logits = logits[:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=-1)
        cur_length += 1

    return input_ids
```

اس بات کو یقینی بنانے کے لیے ذیل میں تجویز کردہ طریقوں پر عمل کریں کہ آپ کا حسب ضرورت جنریشن طریقہ توقع کے مطابق کام کرتا ہے۔

- اصل [`~GenerationMixin.generate`] میں توثیق اور ان پٹ کی تیاری کے لیے منطق کو دوبارہ استعمال کرنے کے لیے آزاد محسوس کریں۔
- اگر آپ `model` میں کوئی نجی طریقہ/انتساب استعمال کرتے ہیں تو ضروریات میں `transformers` ورژن کو پن کریں۔
- ماڈل توثیق، ان پٹ کی توثیق، یا یہاں تک کہ ایک علیحدہ ٹیسٹ فائل شامل کرنے پر غور کریں تاکہ صارفین کو ان کے ماحول میں اپنے کوڈ کی جانچ پڑتال کرنے میں مدد ملے۔

آپ کا حسب ضرورت `generate` طریقہ `custom_generate` فولڈر سے متعلقہ درآمدی کوڈ کو مل سکتا ہے۔ مثال کے طور پر، اگر آپ کے پاس `utils.py` فائل ہے، تو آپ اسے اس طرح درآمد کر سکتے ہیں:

```py
from .utils import some_function
```

ایک ہی سطح کے `custom_generate` فولڈر سے صرف متعلقہ درآمدات ہی معاون ہیں۔ والدین/بہن بھائی فولڈر کی درآمدات درست نہیں ہیں۔ `custom_generate` دلیل مقامی طور پر کسی بھی ڈائریکٹری کے ساتھ کام کرتی ہے جس میں `custom_generate` ڈھانچہ ہوتا ہے، جو آپ کے حسب ضرورت جنریشن کے طریقہ کار کو تیار کرنے کے لیے تجویز کردہ ورک فلو ہے۔

```py
gen_out = model.generate(**inputs, custom_generate="path/to/local/dir", trust_remote_code=True)
```

> [!انتباہ] > مقامی ڈائرکٹری کو لوڈ کرنا اب بھی اس کے `custom_generate/generate.py` پر عمل کرتا ہے، لہذا اسے حب کے ذخیرے کی طرح `trust_remote_code=True` کی ضرورت ہوتی ہے۔ صرف اس کوڈ کے لیے `trust_remote_code=True` پاس کریں جس کا آپ نے لکھا یا جائزہ لیا ہے۔

#### requirements.txt

آپ اختیاری طور پر `custom_generate` فولڈر کے اندر موجود `requirements.txt` فائل میں ازگر کی اضافی ضروریات کی وضاحت کر سکتے ہیں۔ یہ رن ٹائم کے وقت چیک کیے جاتے ہیں اور اگر وہ غائب ہیں تو ایک استثناء دیا جائے گا، صارفین کو اس کے مطابق اپنے ماحول کو اپ ڈیٹ کرنے کے لیے جھنجھوڑا جائے گا۔

#### README.md

ماڈل ریپوزٹری میں روٹ لیول `README.md` عام طور پر اس میں ماڈل کو بیان کرتا ہے۔ تاہم، چونکہ ذخیرے کا فوکس کسٹم جنریشن کا طریقہ ہے، اس لیے ہم اپنی توجہ اپنی مرضی کے مطابق جنریشن کے طریقہ کار کو بیان کرنے کی طرف منتقل کرنے کی انتہائی سفارش کرتے ہیں۔ طریقہ کار کی وضاحت کے علاوہ، ہم تجویز کرتے ہیں کہ اصل [`~GenerationMixin.generate`] میں کسی بھی ان پٹ اور/یا آؤٹ پٹ فرق کو دستاویز کریں۔ اس طرح، صارفین نئی چیزوں پر توجہ مرکوز کر سکتے ہیں، اور عمل درآمد کی عمومی تفصیلات کے لیے Transformers دستاویزات پر انحصار کر سکتے ہیں۔

دریافت کرنے کے لیے، ہم آپ کو اپنے ذخیرے میں `custom_generate` ٹیگ شامل کرنے کا مشورہ دیتے ہیں۔ ایسا کرنے کے لیے، آپ کی `README.md` فائل کا اوپری حصہ نیچے دی گئی مثال کی طرح نظر آنا چاہیے۔ فائل کو دبانے کے بعد، آپ کو اپنے ذخیرے میں ٹیگ نظر آنا چاہیے!

```text
---
library_name: transformers
tags:
  - custom_generate
---

(your markdown content here)
```

تجویز کردہ مشقیں:

- [`~GenerationMixin.generate`] میں دستاویز کے ان پٹ اور آؤٹ پٹ میں فرق۔
- فوری تجربہ کو فعال کرنے کے لیے خود ساختہ مثالیں شامل کریں۔
- نرم تقاضوں کی وضاحت کریں جیسے کہ اگر یہ طریقہ صرف ماڈلز کے مخصوص خاندان کے ساتھ اچھا کام کرتا ہے۔

### `generate` کی ان پٹ تیاری کو دوبارہ استعمال کرنا

اگر آپ ایک نیا ڈی کوڈنگ لوپ شامل کر رہے ہیں، تو آپ `generate` میں موجود ان پٹ تیاری کو محفوظ رکھنا چاہتے ہیں (بیچ کی توسیع، توجہ کے ماسک، لاگٹس پروسیسر، روکنے کے معیار وغیرہ)۔ آپ [`~GenerationMixin.generate`] کی مکمل تیاری کی پائپ لائن کو دوبارہ استعمال کرنے کے لیے **کالبل** کو `custom_generate` پر بھی پاس کر سکتے ہیں جبکہ صرف ڈی کوڈنگ لوپ کو اوور رائیڈ کر سکتے ہیں۔

```py
def custom_loop(model, input_ids, attention_mask, logits_processor, stopping_criteria, generation_config, **model_kwargs):
    next_tokens = input_ids
    while input_ids.shape[1] < stopping_criteria[0].max_length:
        logits = model(next_tokens, attention_mask=attention_mask, **model_kwargs).logits
        next_token_logits = logits_processor(input_ids, logits[:, -1, :])
        next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
        input_ids = torch.cat((input_ids, next_tokens), dim=-1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(next_tokens)), dim=-1)
    return input_ids

output = model.generate(
    **inputs,
    custom_generate=custom_loop,
    max_new_tokens=10,
)
```

> [!ٹپ] > اگر آپ شائع کرتے ہیں۔`custom_generate`ذخیرہ، آپ کا`generate`نفاذ خود ایک کال ایبل کی وضاحت کرسکتا ہے اور اسے منتقل کرسکتا ہے۔`model.generate()`. یہ آپ کو ڈی کوڈنگ لوپ کو اپنی مرضی کے مطابق کرنے دیتا ہے جبکہ اس سے بھی فائدہ ہوتا ہے۔Transformersبلٹ ان پٹ تیاری منطق۔

### کسٹم جنریشن کے طریقے تلاش کرنا

آپ اپنی مرضی کے تمام طریقے تلاش کر سکتے ہیں [ان کے حسب ضرورت ٹیگ کو تلاش کر کے۔](https://huggingface.co/models?other=custom_generate), `custom_generate`۔ ٹیگ کے علاوہ، ہم `custom_generate` طریقوں کے دو مجموعے تیار کرتے ہیں:

- [کسٹم جنریشن کے طریقے - Community](https://huggingface.co/collections/transformers-community/custom-generation-methods-community-6888fb1da0efbc592d3a8ab6) - کمیونٹی کی طرف سے تعاون کردہ طاقتور طریقوں کا مجموعہ؛
- [کسٹم جنریشن کے طریقے - ٹیوٹوریلز](https://huggingface.co/collections/transformers-community/custom-generation-methods-tutorials-6823589657a94940ea02cfec) -- ان طریقوں کے لیے حوالہ جات کے نفاذ کا مجموعہ جو پہلے `transformers` کا حصہ تھے، نیز `custom_generate` کے لیے سبق۔

## وسائل

عام ضابطہ کشائی کی حکمت عملی کس طرح کام کرتی ہے اس کی وضاحت کے لیے [متن کو کیسے تیار کریں: Transformers](https://huggingface.co/blog/how-to-generate) بلاگ پوسٹ کے ساتھ زبان کی تخلیق کے لیے مختلف ضابطہ کشائی کے طریقے استعمال کرتے ہوئے پڑھیں۔
