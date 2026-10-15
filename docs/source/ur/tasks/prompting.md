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

# پرامپٹ انجینئرنگ

[[open-in-colab]]

پرامپٹ انجینئرنگ یا پرامپٹ، مختلف کاموں پر بڑی زبان ماڈل (LLM) کی کارکردگی کو بہتر بنانے کے لیے قدرتی زبان کا استعمال کرتا ہے۔ ایک پرامپٹ ماڈل کو مطلوبہ آؤٹ پٹ پیدا کرنے کی طرف لے جا سکتا ہے۔ بہت سے معاملات میں، آپ کو کسی کام کے لیے [fine-tuned](#fine-tuning) ماڈل کی ضرورت بھی نہیں ہوتی ہے۔ آپ کو صرف ایک اچھا اشارہ درکار ہے۔

LLM کو کچھ متن کی درجہ بندی کرنے کا اشارہ کرنے کی کوشش کریں۔ جب آپ پرامپٹ بناتے ہیں، تو یہ ضروری ہے کہ کام کے بارے میں خاص ہدایات فراہم کریں اور اس کا نتیجہ کیسا ہونا چاہیے۔

```py
from transformers import pipeline
import torch

pipeline = pipeline(task="text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Classify the text into neutral, negative or positive.
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
"""

outputs = pipeline(prompt, max_new_tokens=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Classify the text into neutral, negative or positive. 
Text: This movie is definitely one of my favorite movies of its kind. The interaction between respectable and morally strong characters is an ode to chivalry and the honor code amongst thieves and policemen.
Sentiment:
Positive
```

چیلنج پرامپٹس کو ڈیزائن کرنے میں ہے جو وہ نتائج پیدا کرتا ہے جس کی آپ توقع کر رہے ہیں کیونکہ زبان ناقابل یقین حد تک نازک اور تاثراتی ہے۔

اس گائیڈ میں فوری طور پر انجینئرنگ کے بہترین طریقوں، تکنیکوں اور زبان اور استدلال کے کاموں کو حل کرنے کی مثالیں شامل ہیں۔

## بہترین طریقے

1. بہترین کارکردگی کے لیے تازہ ترین ماڈلز چننے کی کوشش کریں۔ ذہن میں رکھیں کہ LLMs دو قسموں میں آسکتے ہیں، [base](https://hf.co/mistralai/Mistral-7B-v0.1) اور [instruction-tuned](https://hf.co/mistralai/Mistral-7B-Instruct-v0.1) (یا chat)۔

    Base models are excellent at completing text given an initial prompt, but they're not as good at following instructions. Instruction-tuned models are specifically trained versions of the base models on instructional or conversational data. This makes instruction-tuned models a better fit for prompting.

    > [!WARNING]
    > Modern LLMs are typically decoder-only models, but there are some encoder-decoder LLMs like [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5) or [BART](https://huggingface.co/docs/transformers/main/en/model_doc/bart) that may be used for prompting. Load these models directly with the [`AutoModelForSeq2SeqLM`] class (instead of using [`Pipeline`]) and generate outputs from the model itself.

2. ایک مختصر اور آسان اشارے سے شروع کریں، اور بہتر نتائج حاصل کرنے کے لیے اس پر اعادہ کریں۔

3. پرامپٹ کے شروع یا آخر میں ہدایات لگائیں۔ لمبے پرامپٹس کے لیے، ماڈلز توجہ کو چوکور پیمانے پر پیمانہ ہونے سے روکنے کے لیے اصلاح کا اطلاق کر سکتا ہے، جو پرامپٹ کے شروع اور آخر میں زیادہ زور دیتا ہے۔

4. دلچسپی کے متن سے واضح طور پر ہدایات کو الگ کریں۔

5. کام اور مطلوبہ آؤٹ پٹ کے بارے میں مخصوص اور وضاحتی بنیں، بشمول مثال کے طور پر، اس کی شکل، لمبائی، انداز اور زبان۔ مبہم وضاحتوں اور ہدایات سے گریز کریں۔

6. ہدایات کو "کیا نہیں کرنا ہے" کے بجائے "کیا کرنا ہے" پر توجہ مرکوز کرنی چاہئے۔

7. پہلا لفظ یا یہاں تک کہ پہلا جملہ لکھ کر درست آؤٹ پٹ پیدا کرنے کے لیے ماڈل کی قیادت کریں۔

8. نتائج کو بہتر بنانے کے لیے دیگر تکنیکوں جیسے [few-shot](#few-shot-prompting) اور [chain-of-thought](#chain-of-thought) آزمائیں۔

9. اپنے پرامپٹس کی مضبوطی کا اندازہ لگانے کے لیے مختلف ماڈلز کے ساتھ ٹیسٹ کریں۔

10. ورژن اور اپنی فوری کارکردگی کو ٹریک کریں۔

## تکنیک

اکیلے ایک اچھا پرامپٹ تیار کرنا، جسے زیرو شاٹ پرامپٹ بھی کہا جاتا ہے، ہو سکتا ہے کہ آپ مطلوبہ نتائج حاصل کرنے کے لیے کافی نہ ہوں۔ بہترین کارکردگی حاصل کرنے کے لیے آپ کو چند اشتعال انگیز تکنیکوں کو آزمانے کی ضرورت پڑ سکتی ہے۔

یہ سیکشن چند حوصلہ افزا تکنیکوں کا احاطہ کرتا ہے۔

<a id='few-shot-prompting'></a>
### چند شاٹ پرامپٹنگ

چند شاٹ پرامپٹنگ اس بات کی مخصوص مثالیں شامل کرکے درستگی اور کارکردگی کو بہتر بناتی ہے کہ ماڈل کو ان پٹ کے بعد کیا پیدا کرنا چاہیے۔ واضح مثالیں ماڈل کو اس کام اور آؤٹ پٹ فارمیٹ کی بہتر تفہیم فراہم کرتی ہیں جس کی آپ تلاش کر رہے ہیں۔ یہ دیکھنے کے لیے کہ یہ کارکردگی کو کیسے متاثر کرتی ہے، مختلف نمبروں کی مثالوں (2، 4، 8، وغیرہ) کے ساتھ تجربہ کرنے کی کوشش کریں۔ نیچے دی گئی مثال ماڈل کو آؤٹ پٹ فارمیٹ کی 1 مثال (1-شاٹ) کے ساتھ فراہم کرتی ہے (MM/DD/YYYY فارمیٹ میں ایک تاریخ) اسے واپس آنا چاہیے۔

```python
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Text: The first human went into space and orbited the Earth on April 12, 1961.
Date: 04/12/1961
Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon.
Date:"""

outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
# Result: Text: The first human went into space and orbited the Earth on April 12, 1961.
# Date: 04/12/1961
# Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon.
# Date: 09/28/1960
```

چند شاٹ پرامپٹنگ کا منفی پہلو یہ ہے کہ آپ کو لمبے لمبے پرامپٹس بنانے کی ضرورت ہے جس سے حساب اور تاخیر میں اضافہ ہوتا ہے۔ فوری طوالت کی بھی ایک حد ہوتی ہے۔ آخر میں، ایک ماڈل آپ کی مثالوں سے غیر ارادی نمونے سیکھ سکتا ہے، اور ہو سکتا ہے کہ یہ پیچیدہ استدلال کے کاموں پر اچھی طرح کام نہ کرے۔

جدید انسٹرکشن ٹیونڈ LLMs کے لیے چند شاٹ پرامپٹنگ کو بہتر بنانے کے لیے، ماڈل کا مخصوص [chat template](https://huggingface.co/docs/transformers/main/en/conversations) استعمال کریں۔ یہ ماڈلز ڈیٹا سیٹس پر تربیت یافتہ ہیں جن میں "صارف" اور "اسسٹنٹ" کے درمیان باری پر مبنی گفتگو ہوتی ہے۔ اس کے ساتھ سیدھ میں لانے کے لیے اپنے پرامپٹ کی تشکیل کارکردگی کو بہتر بنا سکتی ہے۔

اپنے پرامپٹ کو موڑ پر مبنی گفتگو کے طور پر تشکیل دیں اور اسے ٹوکنائز کرنے اور فارمیٹ کرنے کے لیے [`apply_chat_template`] طریقہ استعمال کریں۔

```python
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")

messages = [
    {"role": "user", "content": "Text: The first human went into space and orbited the Earth on April 12, 1961."},
    {"role": "assistant", "content": "Date: 04/12/1961"},
    {"role": "user", "content": "Text: The first-ever televised presidential debate in the United States took place on September 28, 1960, between presidential candidates John F. Kennedy and Richard Nixon."}
]

prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = pipeline(prompt, max_new_tokens=12, do_sample=True, top_k=10)

for output in outputs:
    print(f"Result: {output['generated_text']}")
```

اگرچہ بنیادی چند شاٹ پرامپٹنگ اپروچ ایک ہی ٹیکسٹ سٹرنگ میں مثالوں کو سرایت کرتا ہے، چیٹ ٹیمپلیٹ فارمیٹ درج ذیل فوائد پیش کرتا ہے۔

- ماڈل میں ممکنہ طور پر بہتر تفہیم ہوسکتی ہے کیونکہ یہ پیٹرن اور صارف کے ان پٹ اور اسسٹنٹ آؤٹ پٹ کے متوقع کردار کو بہتر طریقے سے پہچان سکتا ہے۔
- ماڈل مطلوبہ آؤٹ پٹ فارمیٹ کو زیادہ مستقل طور پر آؤٹ پٹ کر سکتا ہے کیونکہ اس کی ساخت تربیت کے دوران اس کے ان پٹ کی طرح ہوتی ہے۔

ان کے چیٹ ٹیمپلیٹ کے فارمیٹ کے بارے میں مزید جاننے کے لیے ہمیشہ مخصوص ہدایات پر مبنی ماڈل کی دستاویزات سے مشورہ کریں تاکہ آپ اس کے مطابق اپنے چند شاٹ پرامپٹس کو تشکیل دے سکیں۔

<a id='chain-of-thought'></a>
### سوچ کا سلسلہ

چین آف تھاٹ (CoT) پرامپٹس کی ایک سیریز فراہم کرکے زیادہ مربوط اور معقول نتائج پیدا کرنے میں موثر ہے جو کسی موضوع کے بارے میں ماڈل "سوچنے" میں مدد کرتے ہیں۔

ذیل کی مثال ماڈل فراہم کرتی ہے جس میں درمیانی استدلال کے مراحل کے ذریعے کام کرنے کے لیے کئی اشارے ہیں۔

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Let's go through this step-by-step:
1. You start with 15 muffins.
2. You eat 2 muffins, leaving you with 13 muffins.
3. You give 5 muffins to your neighbor, leaving you with 8 muffins.
4. Your partner buys 6 more muffins, bringing the total number of muffins to 14.
5. Your partner eats 2 muffins, leaving you with 12 muffins.
If you eat 6 muffins, how many are left?
Answer: 6
```

جیسے [few-shot](#few-shot-prompting) prompting، CoT کا منفی پہلو یہ ہے کہ اسے پرامپٹس کی ایک سیریز ڈیزائن کرنے کے لیے زیادہ محنت کی ضرورت ہوتی ہے جو ماڈل وجہ کو ایک پیچیدہ کام کے ذریعے مدد کرتے ہیں اور فوری لمبائی تاخیر کو بڑھاتی ہے۔

<a id='fine-tuning'></a>
## فائن ٹیوننگ

LLMs کے ساتھ کام کرنے کے لیے پرامپٹ کرنا ایک طاقتور طریقہ ہے، لیکن ایسے منظرنامے ہیں جہاں ماڈل یا یہاں تک کہ فائن ٹیوننگ a ماڈل بہتر کام کرتا ہے۔

یہاں کچھ مثالوں کے منظرنامے ہیں جہاں ایک عمدہ ماڈل معنی رکھتا ہے۔

- آپ کا ڈومین اس سے بالکل مختلف ہے جس پر LLM پہلے سے تربیت یافتہ تھا، اور وسیع تر اشارے سے آپ کے مطلوبہ نتائج برآمد نہیں ہوئے۔
- آپ کے ماڈل کو کم وسائل والی زبان میں اچھی طرح سے کام کرنے کی ضرورت ہے۔
- آپ کے ماڈل کو حساس ڈیٹا کے بارے میں تربیت دینے کی ضرورت ہے جس کے سخت ضابطے کے تقاضے ہیں۔
- آپ لاگت، رازداری، انفراسٹرکچر، یا دیگر رکاوٹوں کی وجہ سے ایک چھوٹا ماڈل استعمال کر رہے ہیں۔

ان تمام منظرناموں میں، اس بات کو یقینی بنائیں کہ آپ کے پاس اپنے ماڈل کو تربیت دینے کے لیے کافی بڑا ڈومین مخصوص ڈیٹا سیٹ ہے، آپ کے پاس کافی وقت اور وسائل ہیں، اور فائن ٹیوننگ کی قیمت اس کے قابل ہے۔ بصورت دیگر، آپ اپنے پرامپٹ کو بہتر بنانے کی کوشش کرنے سے بہتر ہو سکتے ہیں۔

## مثالیں

ذیل کی مثالیں مختلف کاموں کے لیے LLM کا اشارہ کرتی ہیں۔

<hfoptions id="tasks">
<hfoption id="named entity recognition">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Return a list of named entities in the text.
Text: The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers.
Named entities:
"""

outputs = pipeline(prompt, max_new_tokens=50, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result:  [Clément Delangue, Julien Chaumond, Thomas Wolf, company, New York City, chatbot app, teenagers]
```

</hfoption>
<hfoption id="translation">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Translate the English text to French.
Text: Sometimes, I've believed as many as six impossible things before breakfast.
Translation:
"""

outputs = pipeline(prompt, max_new_tokens=20, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: À l'occasion, j'ai croyu plus de six choses impossibles
```

</hfoption>
<hfoption id="summarization">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Permaculture is a design process mimicking the diversity, functionality and resilience of natural ecosystems. The principles and practices are drawn from traditional ecological knowledge of indigenous cultures combined with modern scientific understanding and technological innovations. Permaculture design provides a framework helping individuals and communities develop innovative, creative and effective strategies for meeting basic needs while preparing for and mitigating the projected impacts of climate change.
Write a summary of the above text.
Summary:
"""

outputs = pipeline(prompt, max_new_tokens=30, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: Permaculture is the design process that involves mimicking natural ecosystems to provide sustainable solutions to basic needs. It is a holistic approach that comb
```

</hfoption>
<hfoption id="question answering">

```py
from transformers import pipeline
import torch

pipeline = pipeline(model="mistralai/Mistral-7B-Instruct-v0.1", dtype=torch.bfloat16, device_map="auto")
prompt = """Answer the question using the context below.
Context: Gazpacho is a cold soup and drink made of raw, blended vegetables. Most gazpacho includes stale bread, tomato, cucumbers, onion, bell peppers, garlic, olive oil, wine vinegar, water, and salt. Northern recipes often include cumin and/or pimentón (smoked sweet paprika). Traditionally, gazpacho was made by pounding the vegetables in a mortar with a pestle; this more laborious method is still sometimes used as it helps keep the gazpacho cool and avoids the foam and silky consistency of smoothie versions made in blenders or food processors.
Question: What modern tool is used to make gazpacho?
Answer:
"""

outputs = pipeline(prompt, max_new_tokens=10, do_sample=True, top_k=10, return_full_text=False)
for output in outputs:
    print(f"Result: {output['generated_text']}")
Result: A blender or food processor is the modern tool
```

</hfoption>
</hfoptions>
