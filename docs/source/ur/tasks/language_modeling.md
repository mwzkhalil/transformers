<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# زبان کی کازل ماڈلنگ

[[open-in-colab]]

زبان کی ماڈلنگ کی دو قسمیں ہیں، causal اور ماسکڈ۔ یہ گائیڈ کازل لینگویج ماڈلنگ کی وضاحت کرتا ہے۔ متن کی تخلیق کے لیے کارآمد زبان ماڈلز کثرت سے استعمال ہوتی ہے۔ آپ ان ماڈلز کو تخلیقی ایپلی کیشنز کے لیے استعمال کر سکتے ہیں جیسے کہ اپنا ٹیکسٹ ایڈونچر منتخب کرنا یا ایک ذہین کوڈنگ اسسٹنٹ جیسے Copilot یا CodeParrot۔

<Youtube id="Vpjb1lu0MDk"/>

کازل لینگویج ماڈلنگ ٹوکنز کی ایک ترتیب میں اگلے ٹوکن کی پیشین گوئی کرتی ہے، اور ماڈل صرف بائیں جانب والے ٹوکنز میں شرکت کر سکتا ہے۔ اس کا مطلب ہے کہ ماڈل مستقبل کے ٹوکن نہیں دیکھ سکتا۔ GPT-2 ایک causal زبان ماڈل کی ایک مثال ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [ELI5](https://huggingface.co/distilbert/distilgpt2) ڈیٹاسیٹ کے [r/askscience](https://www.reddit.com/r/askscience/) سب سیٹ پر DistilGPT2](https://huggingface.co/datasets/dany0407/eli5_category)۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/text-generation) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## ELI5 ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری کے ساتھ [ELI5-Category](https://huggingface.co/datasets/dany0407/eli5_category) ڈیٹاسیٹ سے پہلی 5000 مثالیں لوڈ کرکے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("dany0407/eli5_category", split="train[:5000]")
```

ڈیٹا سیٹ کے `train` کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
  'text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
   'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
   'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
   'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
  'score': [21, 19, 5, 3],
  'text_urls': [[],
   [],
   [],
   ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

اگرچہ یہ بہت زیادہ لگ سکتا ہے، آپ کی دلچسپی صرف `text` فیلڈ میں ہے۔ لینگویج ماڈلنگ کے کاموں کے بارے میں سب سے اچھی بات یہ ہے کہ آپ کو لیبلز کی ضرورت نہیں ہے (جسے ایک غیر نگرانی شدہ کام بھی کہا جاتا ہے) کیونکہ اگلا لفظ *لیبل* ہے۔

## پری پروسیس

<Youtube id="ma1TrR7gE7I"/>

اگلا مرحلہ `text` ذیلی فیلڈ پر کارروائی کرنے کے لیے DistilGPT2 ٹوکنائزر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
```

آپ اوپر دی گئی مثال سے دیکھیں گے، `text` فیلڈ دراصل `answers` کے اندر اندر بنا ہوا ہے۔ اس کا مطلب ہے کہ آپ کو [`flatten`](https://huggingface.co/docs/datasets/process#flatten) طریقہ کے ساتھ `text` ذیلی فیلڈ کو اس کے نیسٹڈ ڈھانچے سے نکالنے کی ضرورت ہوگی:

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
 'title': 'What does the tax bill that was passed today mean? How will it affect Americans in each tax bracket?',
 'selftext': '',
 'category': 'Economics',
 'subreddit': 'explainlikeimfive',
 'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
 'answers.text': ["The tax bill is 500 pages long and there were a lot of changes still going on right to the end. It's not just an adjustment to the income tax brackets, it's a whole bunch of changes. As such there is no good answer to your question. The big take aways are: - Big reduction in corporate income tax rate will make large companies very happy. - Pass through rate change will make certain styles of business (law firms, hedge funds) extremely happy - Income tax changes are moderate, and are set to expire (though it's the kind of thing that might just always get re-applied without being made permanent) - People in high tax states (California, New York) lose out, and many of them will end up with their taxes raised.",
  'None yet. It has to be reconciled with a vastly different house bill and then passed again.',
  'Also: does this apply to 2017 taxes? Or does it start with 2018 taxes?',
  'This article explains both the House and senate bills, including the proposed changes to your income taxes based on your income level. URL_0'],
 'answers.score': [21, 19, 5, 3],
 'answers.text_urls': [[],
  [],
  [],
  ['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
 'title_urls': ['url'],
 'selftext_urls': ['url']}
```

ہر ذیلی فیلڈ اب ایک الگ کالم ہے جیسا کہ `answers` سابقہ ​​سے اشارہ کیا گیا ہے، اور `text` فیلڈ اب ایک فہرست ہے۔ ہر جملے کو الگ الگ ٹوکنائز کرنے کے بجائے، فہرست کو سٹرنگ میں تبدیل کریں تاکہ آپ ان کو مشترکہ طور پر ٹوکنائز کر سکیں۔

یہاں ہر ایک مثال کے لیے سٹرنگز کی فہرست میں شامل ہونے اور نتیجہ کو ٹوکنائز کرنے کے لیے پہلا پری پروسیسنگ فنکشن ہے:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

اس پری پروسیسنگ فنکشن کو پورے ڈیٹاسیٹ پر لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] طریقہ استعمال کریں۔ آپ ڈیٹاسیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کرکے اور `num_proc` کے ساتھ عمل کی تعداد بڑھا کر `map` فنکشن کو تیز کر سکتے ہیں۔ ایسے کالموں کو ہٹا دیں جن کی آپ کو ضرورت نہیں ہے:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

اس ڈیٹاسیٹ میں ٹوکن کی ترتیبیں شامل ہیں، لیکن ان میں سے کچھ ماڈل کے لیے زیادہ سے زیادہ ان پٹ کی لمبائی سے زیادہ لمبی ہیں۔

اب آپ دوسرے پری پروسیسنگ فنکشن کو استعمال کرسکتے ہیں۔

- تمام ترتیب کو جوڑیں۔
- مربوط ترتیبوں کو `block_size` کے ذریعہ بیان کردہ چھوٹے حصوں میں تقسیم کریں، جو زیادہ سے زیادہ ان پٹ لمبائی سے کم اور آپ کے GPU RAM کے لیے کافی مختصر ہونا چاہیے۔

```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

پورے ڈیٹاسیٹ پر `group_texts` فنکشن کا اطلاق کریں:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

اب [`DataCollatorForLanguageModeling`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ مجموعہ کے دوران تمام ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے جملوں کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔

ترتیب کے اختتامی ٹوکن کو پیڈنگ ٹوکن کے طور پر استعمال کریں اور `mlm=False` سیٹ کریں۔ یہ ان پٹس کا استعمال کرے گا جیسا کہ لیبل ایک عنصر کے ذریعے دائیں طرف شفٹ ہوتے ہیں:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو [بنیادی ٹیوٹوریل](../training) پر ایک نظر ڈالیں!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForCausalLM`] کے ساتھ DistilGPT2 لوڈ کریں:

```py
>>> from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔
2. ماڈل، ڈیٹا سیٹس، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_clm-model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"],
...     eval_dataset=lm_dataset["test"],
...     data_collator=data_collator,
...     processing_class=tokenizer,
... )

>>> trainer.train()
```

ٹریننگ مکمل ہونے کے بعد، اپنے ماڈل کا اندازہ کرنے اور اس کی پریشانی حاصل کرنے کے لیے [`~transformers.Trainer.evaluate`] طریقہ استعمال کریں:

```py
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 49.61
```

پھر اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب میں بانٹیں تاکہ ہر کوئی آپ کا ماڈل استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

ماڈل کو کازل لینگویج ماڈلنگ کے لیے کیسے ٹھیک کیا جائے اس کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

ایک پرامپٹ لے کر آئیں جس سے آپ متن بنانا چاہیں گے:

```py
>>> prompt = "Somatic hypermutation allows the immune system to"
```

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ ٹیکسٹ جنریشن کے لیے ایک `pipeline` فوری بنائیں، اور اپنا متن اس تک پہنچائیں:

```py
>>> from transformers import pipeline

>>> generator = pipeline("text-generation", model="username/my_awesome_eli5_clm-model")
>>> generator(prompt)
[{'generated_text': "Somatic hypermutation allows the immune system to be able to effectively reverse the damage caused by an infection.\n\n\nThe damage caused by an infection is caused by the immune system's ability to perform its own self-correcting tasks."}]
```

متن کو ٹوکنائز کریں اور `input_ids` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_clm-model")
>>> inputs = tokenizer(prompt, return_tensors="pt").input_ids
```

متن بنانے کے لیے [`~generation.GenerationMixin.generate`] طریقہ استعمال کریں۔ متن جنریشن کی مختلف حکمت عملیوں اور جنریشن کو کنٹرول کرنے کے پیرامیٹرز کے بارے میں مزید تفصیلات کے لیے، [ٹیکسٹ جنریشن کی حکمت عملی](../generation_strategies) صفحہ دیکھیں۔

```py
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("username/my_awesome_eli5_clm-model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

تیار کردہ ٹوکن آئی ڈی کو واپس متن میں ڈی کوڈ کریں:

```py
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["Somatic hypermutation allows the immune system to react to drugs with the ability to adapt to a different environmental situation. In other words, a system of 'hypermutation' can help the immune system to adapt to a different environmental situation or in some cases even a single life. In contrast, researchers at the University of Massachusetts-Boston have found that 'hypermutation' is much stronger in mice than in humans but can be found in humans, and that it's not completely unknown to the immune system. A study on how the immune system"]
```
