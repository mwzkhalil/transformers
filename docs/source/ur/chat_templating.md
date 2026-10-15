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

# چیٹ ٹیمپلیٹس

[چیٹ کی بنیادی باتیں](https://huggingface.co/docs/transformers/main/en/conversations) گائیڈ میں [`TextGenerationPipeline`] کا استعمال کرتے ہوئے چیٹ کی تاریخوں کو ذخیرہ کرنے اور چیٹ ماڈلز سے متن بنانے کا طریقہ شامل ہے۔

یہ گائیڈ زیادہ جدید صارفین کے لیے ہے، اور اس میں بنیادی کلاسز اور طریقوں کے ساتھ ساتھ یہ سمجھنے کے لیے کلیدی تصورات کا احاطہ کیا گیا ہے کہ جب آپ ماڈل کے ساتھ چیٹ کرتے ہیں تو اصل میں کیا ہو رہا ہے۔

چیٹ کو سمجھنے کے لیے اہم بصیرت کی ضرورت ہے۔ماڈلزیہ ہے: تمام causal LMs، چاہے چیٹ سے تربیت یافتہ ہوں یا نہیں، ٹوکنز کا سلسلہ جاری رکھیں۔ جب causal LMs کو تربیت دی جاتی ہے، تربیت عام طور پر متن کے ایک بڑے کارپس پر "پری ٹریننگ" کے ساتھ شروع ہوتی ہے، جو ایک "بیس" بناتا ہے۔ماڈل. یہ بیسماڈلزاس کے بعد اکثر چیٹ کے لیے "فائن ٹیون" ہوتے ہیں، جس کا مطلب ہے کہ انہیں ڈیٹا پر تربیت دینا جو پیغامات کی ترتیب کے طور پر فارمیٹ کیا جاتا ہے۔ چیٹ ابھی بھی ٹوکن کا ایک سلسلہ ہے، حالانکہ! کی فہرست`role`اور`content`وہ لغات جو آپ چیٹ میں بھیجتے ہیں۔ماڈلایک ٹوکن ترتیب میں تبدیل ہو جائیں، اکثر کنٹرول ٹوکن جیسے`<|user|>`یا`<|assistant|>`یا`<|end_of_message|>`، جو اجازت دیتا ہے۔ماڈلچیٹ کا ڈھانچہ دیکھنے کے لیے۔ بہت سے ممکنہ چیٹ فارمیٹس ہیں، اور مختلفماڈلزمختلف فارمیٹس یا کنٹرول ٹوکن استعمال کر سکتے ہیں، چاہے وہ ایک ہی بنیاد سے ٹھیک بنائے گئے ہوں۔ماڈل!

گھبرائیں نہیں، اگرچہ - آپ کو چیٹ ماڈلز استعمال کرنے کے لیے ہر ممکنہ چیٹ فارمیٹ کو یاد کرنے کی ضرورت نہیں ہے۔ چیٹ ماڈلز **چیٹ ٹیمپلیٹس** کے ساتھ آتے ہیں، جو اس بات کی نشاندہی کرتے ہیں کہ وہ چیٹس کی فارمیٹ ہونے کی توقع کیسے کرتے ہیں۔ آپ [`apply_chat_template`] طریقہ سے ان تک رسائی حاصل کر سکتے ہیں۔ آئیے دو مثالیں دیکھتے ہیں۔ یہ دونوں ماڈلز ایک ہی `Mistral-7B` بیس ماڈل سے ٹھیک ہیں:

<hfoptions id="template">
<hfoption id="Mistral">

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
```

```md
<s>[INST] Hello, how are you? [/INST]I'm doing great. How can I help you today?</s> [INST] I'd like to show off how chat templating works! [/INST]
```

</hfoption>
<hfoption id="Zephyr">

```py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
chat = [
  {"role": "user", "content": "Hello, how are you?"},
  {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
  {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

tokenizer.apply_chat_template(chat, tokenize=False)
```

```md
<|user|>\nHello, how are you?</s>\n<|assistant|>\nI'm doing great. How can I help you today?</s>\n<|user|>\nI'd like to show off how chat templating works!</s>\n
```

</hfoption>
</hfoptions>

Mistral-7B-Instruct صارف کے پیغامات کے آغاز اور اختتام کی نشاندہی کرنے کے لیے `[INST]` اور `[/INST]` ٹوکنز کا استعمال کرتا ہے، جبکہ Zephyr-7B اسپیکر کے کردار کی نشاندہی کرنے کے لیے `<|user|>` اور `<|assistant|>` ٹوکن استعمال کرتا ہے۔ یہی وجہ ہے کہ چیٹ ٹیمپلیٹس اہم ہیں - غلط کنٹرول ٹوکنز کے ساتھ، ان ماڈلز کی کارکردگی بہت زیادہ خراب ہوگی۔

## `apply_chat_template` استعمال کرنا

`apply_chat_template` کے ان پٹ کو لغات کی فہرست کے طور پر `role` اور `content` کلیدوں کے ساتھ تشکیل دیا جانا چاہیے۔ `role` کلید اسپیکر کی وضاحت کرتی ہے، اور `content` کلید پیغام پر مشتمل ہے۔ عام کردار ہیں:

- صارف کے پیغامات کے لیے `user`
- `assistant` سے پیغامات کے لیے ماڈل
- `system` ہدایات کے لیے کہ ماڈل کو کیسے کام کرنا چاہیے (عام طور پر چیٹ کے آغاز میں رکھا جاتا ہے)

[`apply_chat_template`] اس فہرست کو لیتا ہے اور فارمیٹ شدہ ترتیب لوٹاتا ہے۔ اگر آپ ترتیب کو ٹوکنائز کرنا چاہتے ہیں تو `tokenize=True` سیٹ کریں۔

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceH4/zephyr-7b-beta", device_map="auto", dtype=torch.bfloat16)

messages = [
    {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tokenizer.decode(tokenized_chat["input_ids"][0]))
```

```md
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
```

جواب دینے کے لیے ٹوکنائزڈ چیٹ کو [`~GenerationMixin.generate`] پر منتقل کریں۔

```py
outputs = model.generate(**tokenized_chat, max_new_tokens=128)
print(tokenizer.decode(outputs[0]))
```

```md
<|system|>
You are a friendly chatbot who always responds in the style of a pirate</s>
<|user|>
How many helicopters can a human eat in one sitting?</s>
<|assistant|>
Matey, I'm afraid I must inform ye that humans cannot eat helicopters. Helicopters are not food, they are flying machines. Food is meant to be eaten, like a hearty plate o' grog, a savory bowl o' stew, or a delicious loaf o' bread. But helicopters, they be for transportin' and movin' around, not for eatin'. So, I'd say none, me hearties. None at all.
```

> [!انتباہ] > کچھ ٹوکنائزرز خصوصی `<bos>` اور `<eos>` ٹوکنز شامل کرتے ہیں۔ چیٹ ٹیمپلیٹس میں پہلے سے ہی تمام ضروری خصوصی ٹوکنز شامل ہونے چاہئیں، اور اضافی خصوصی ٹوکنز شامل کرنا اکثر غلط یا ڈپلیکیٹ ہوتا ہے، جس سے ماڈل کارکردگی کو نقصان پہنچتا ہے۔ جب آپ متن کو `apply_chat_template(tokenize=False)` کے ساتھ فارمیٹ کرتے ہیں، تو یقینی بنائیں کہ آپ نے `add_special_tokens=False` سیٹ کیا ہے اگر آپ بعد میں ٹوکنائز کرتے ہیں تاکہ ان ٹوکنز کو نقل کرنے سے بچ سکیں۔ > اگر آپ `apply_chat_template(tokenize=True)` استعمال کرتے ہیں تو یہ کوئی مسئلہ نہیں ہے، جس کا مطلب ہے کہ یہ عام طور پر محفوظ آپشن ہے!

### add_generation_prompt

آپ نے اوپر دی گئی مثالوں میں [`~PreTrainedTokenizerBase.apply_chat_template#add_generation_prompt`] دلیل کو دیکھا ہوگا۔ یہ دلیل چیٹ کے اختتام پر ٹوکنز کا اضافہ کرتی ہے جو `assistant` جواب کے آغاز کی نشاندہی کرتی ہے۔ یاد رکھیں: چیٹ کے تمام تجریدات کے نیچے، چیٹ ماڈلز اب بھی صرف زبان ماڈلز ہیں جو ٹوکنز کا سلسلہ جاری رکھتی ہیں! اگر آپ ایسے ٹوکنز شامل کرتے ہیں جو بتاتے ہیں کہ یہ اب `assistant` جواب میں ہے، تو یہ صحیح طریقے سے جواب لکھے گا، لیکن اگر آپ ان ٹوکنز کو شامل نہیں کرتے ہیں، تو ماڈل الجھن میں پڑ سکتا ہے اور کچھ عجیب کر سکتا ہے، جیسے **جاری رکھنا** صارف کے پیغام کا جواب دینے کے بجائے!

آئیے یہ سمجھنے کے لیے ایک مثال دیکھتے ہیں کہ `add_generation_prompt` اصل میں کیا کر رہا ہے۔ سب سے پہلے، آئیے بغیر `add_generation_prompt` کے چیٹ کو فارمیٹ کریں:

```py
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
tokenized_chat
```

```md
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
```

اب، آئیے اسی چیٹ کو `add_generation_prompt=True` کے ساتھ فارمیٹ کریں:

```py
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
tokenized_chat
```

```md
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant

```

جب `add_generation_prompt=True`، `assistant` پیغام کے آغاز کی نشاندہی کرنے کے لیے آخر میں `<|im_start|>assistant` شامل کیا جاتا ہے۔ یہ ماڈل کو جاننے دیتا ہے کہ `assistant` جواب اگلا ہے۔

تمام ماڈلز جنریشن پرامپٹس کی ضرورت نہیں ہے، اور کچھ ماڈلز، جیسے [Llama](https://huggingface.co/docs/transformers/main/en/model_doc/llama)، `assistant` جواب سے پہلے کوئی خاص ٹوکن نہیں رکھتے ہیں۔ ان صورتوں میں، [`~PreTrainedTokenizerBase.apply_chat_template#add_generation_prompt`] کا کوئی اثر نہیں ہوتا ہے۔

### جاری_فائنل_پیغام

[`~PreTrainedTokenizerBase.apply_chat_template#continue_final_message`] پیرامیٹر کنٹرول کرتا ہے کہ چیٹ میں حتمی پیغام کو نیا شروع کرنے کے بجائے جاری رکھا جائے یا نہیں۔ یہ ترتیب کے اختتامی ٹوکنز کو ہٹاتا ہے تاکہ ماڈل حتمی پیغام سے نسل جاری رکھے۔

یہ ماڈل جواب کو "پہلے سے بھرنے" کے لیے مفید ہے۔ نیچے دی گئی مثال میں، ماڈل متن تیار کرتا ہے جو نیا پیغام شروع کرنے کے بجائے JSON سٹرنگ کو جاری رکھتا ہے۔ یہ مندرجہ ذیل ہدایات کی درستگی کو بہتر بنانے کے لیے بہت مفید ہو سکتا ہے جب آپ جانتے ہوں کہ اس کے جوابات کیسے شروع کیے جائیں۔

```py
chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=True, return_dict=True, continue_final_message=True)
model.generate(**formatted_chat)
```

> [!انتباہ] > آپ کو [`~PreTrainedTokenizerBase.apply_chat_template#add_generation_prompt`] اور [`~PreTrainedTokenizerBase.apply_chat_template#continue_final_message`] ایک ساتھ استعمال نہیں کرنا چاہیے۔ سابقہ ​​ٹوکنز جوڑتا ہے جو ایک نیا پیغام شروع کرتے ہیں، جبکہ مؤخر الذکر ترتیب ٹوکن کے اختتام کو ہٹاتا ہے۔ ان کو ایک ساتھ استعمال کرنے سے غلطی واپس آتی ہے۔

`content` کے بجائے اس فیلڈ کو پہلے سے بھرنے کے لیے ایک فیلڈ کا نام سٹرنگ کے طور پر پاس کریں۔ استدلال ماڈلز اکثر ایک الگ فیلڈ کو ظاہر کرتا ہے، جیسے Qwen پر `reasoning_content` یا Gemma پر `thinking`۔ پری فلنگ `content` جنریشن شروع ہونے سے پہلے ریجننگ بلاک کو بند کر دیتی ہے، اس لیے ماڈل اس کے اندر جاری نہیں رہ سکتا۔ استدلال کے میدان کو پہلے سے بھرنا براہ راست بلاک کو کھلا چھوڑ دیتا ہے۔

```py
chat = [
    {"role": "user", "content": "Explain 1+1"},
    {"role": "assistant", "reasoning_content": "The user wants a simple addition. ", "content": ""},
]

formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, continue_final_message="reasoning_content")
```

حتمی پیغام پر نامزد فیلڈ کا ہونا ضروری ہے اور چیٹ ٹیمپلیٹ کے ذریعہ اس کا حوالہ دیا جانا چاہیے۔ جب یا تو چیک ناکام ہوجاتا ہے تو ایک خرابی پیدا ہوتی ہے۔

[`TextGenerationPipeline`] ایک نیا پیغام شروع کرنے کے لیے بطور ڈیفالٹ [`~PreTrainedTokenizerBase.apply_chat_template#add_generation_prompt`] سے `True` سیٹ کرتا ہے۔ تاہم، اگر چیٹ میں حتمی پیغام کا کردار `assistant` ہے، تو یہ فرض کرتا ہے کہ پیغام پہلے سے بھرا ہوا ہے اور `continue_final_message=True` پر سوئچ ہو جاتا ہے۔ اس کی وجہ یہ ہے کہ زیادہ تر ماڈلز متعدد مسلسل معاون پیغامات کی حمایت نہیں کرتے ہیں۔ اس رویے کو اوور رائیڈ کرنے کے لیے، واضح طور پر [`~PreTrainedTokenizerBase.apply_chat_template#continue_final_message`] دلیل کو پائپ لائن میں منتقل کریں۔

## ماڈل تربیت

چیٹ ٹیمپلیٹ کے ساتھ ماڈل کو تربیت دینا اس بات کو یقینی بنانے کا ایک اچھا طریقہ ہے کہ ٹیمپلیٹ ماڈل کے ٹوکنز سے مماثل ہے۔ چیٹ ٹیمپلیٹ کو اپنے ڈیٹاسیٹ پر پری پروسیسنگ مرحلہ کے طور پر لاگو کریں۔ `add_generation_prompt=False` سیٹ کریں کیونکہ معاون جواب دینے کے لیے اضافی ٹوکن تربیت کے دوران مددگار نہیں ہوتے ہیں۔

چیٹ ٹیمپلیٹ کے ساتھ ڈیٹا سیٹ کو پری پروسیس کرنے کی ایک مثال ذیل میں دکھائی گئی ہے۔

```py
from transformers import AutoTokenizer
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

chat1 = [
    {"role": "user", "content": "Which is bigger, the moon or the sun?"},
    {"role": "assistant", "content": "The sun."}
]
chat2 = [
    {"role": "user", "content": "Which is bigger, a virus or a bacterium?"},
    {"role": "assistant", "content": "A bacterium."}
]

dataset = Dataset.from_dict({"chat": [chat1, chat2]})
dataset = dataset.map(lambda x: {"formatted_chat": tokenizer.apply_chat_template(x["chat"], tokenize=False, add_generation_prompt=False)})
print(dataset['formatted_chat'][0])
```

```md
<|user|>
Which is bigger, the moon or the sun?</s>
<|assistant|>
The sun.</s>
```

اس قدم کے بعد، آپ `formatted_chat` کالم کا استعمال کرتے ہوئے [ٹریننگ ریسیپی](./tasks/language_modeling) causal language ماڈلز کی پیروی جاری رکھ سکتے ہیں۔
