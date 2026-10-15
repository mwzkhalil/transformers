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

# زبان کی ماسکڈ ماڈلنگ

[[open-in-colab]]

<Youtube id="mqElG5QJWUg"/>

نقاب پوش زبان کی ماڈلنگ ایک ترتیب میں ایک نقاب پوش ٹوکن کی پیشین گوئی کرتی ہے، اور ماڈل ٹوکنز پر دو طرفہ طور پر شرکت کر سکتا ہے۔ اس کا مطلب ہے کہ ماڈل کو بائیں اور دائیں طرف کے ٹوکنز تک مکمل رسائی حاصل ہے۔ ماسکڈ لینگویج ماڈلنگ ان کاموں کے لیے بہترین ہے جن کے لیے پورے تسلسل کی اچھی سیاق و سباق کی سمجھ کی ضرورت ہوتی ہے۔ BERT ایک نقاب پوش زبان ماڈل کی ایک مثال ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [ELI5](https://huggingface.co/distilbert/distilroberta-base) ڈیٹاسیٹ کے [r/askscience](https://www.reddit.com/r/askscience/) سب سیٹ پر DistilRoBERta](https://huggingface.co/datasets/dany0407/eli5_category)۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/fill-mask) کو چیک کرنے کی تجویز کرتے ہیں۔

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

<Youtube id="8PmhEIXhBvI"/>

ماسکڈ لینگویج ماڈلنگ کے لیے، اگلا مرحلہ `text` ذیلی فیلڈ پر کارروائی کرنے کے لیے DistilRoBERTA ٹوکنائزر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
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
...     return result
```

پورے ڈیٹاسیٹ پر `group_texts` فنکشن کا اطلاق کریں:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

اب [`DataCollatorForLanguageModeling`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ مجموعہ کے دوران تمام ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے جملوں کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔

ترتیب کے اختتام والے ٹوکن کو پیڈنگ ٹوکن کے طور پر استعمال کریں اور ہر بار جب آپ ڈیٹا پر اعادہ کریں تو ٹوکن کو بے ترتیب طور پر ماسک کرنے کے لیے `mlm_probability` کی وضاحت کریں:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForMaskedLM`] کے ساتھ DistilRoBERta لوڈ کریں:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔
2. ماڈل، ڈیٹا سیٹس، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     num_train_epochs=3,
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
Perplexity: 8.76
```

پھر اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب میں بانٹیں تاکہ ہر کوئی آپ کا ماڈل استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

ماسکڈ لینگویج ماڈلنگ کے لیے ماڈل کو کیسے ٹھیک کیا جائے اس کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

کچھ متن لے کر آئیں جس کے ساتھ آپ خالی کو پُر کرنے کے لیے ماڈل چاہتے ہیں، اور خالی کی نشاندہی کرنے کے لیے خصوصی `<mask>` ٹوکن استعمال کریں:

```py
>>> text = "The Milky Way is a <mask> galaxy."
```

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ فل ماسک کے لیے ایک `pipeline` فوری بنائیں، اور اپنا متن اس تک پہنچائیں۔ اگر آپ چاہیں تو، آپ `top_k` پیرامیٹر استعمال کر سکتے ہیں یہ بتانے کے لیے کہ کتنی پیشین گوئیاں واپس کرنی ہیں:

```py
>>> from transformers import pipeline

>>> mask_filler = pipeline("fill-mask", "username/my_awesome_eli5_mlm_model")
>>> mask_filler(text, top_k=3)
[{'score': 0.5150994658470154,
  'token': 21300,
  'token_str': ' spiral',
  'sequence': 'The Milky Way is a spiral galaxy.'},
 {'score': 0.07087188959121704,
  'token': 2232,
  'token_str': ' massive',
  'sequence': 'The Milky Way is a massive galaxy.'},
 {'score': 0.06434620916843414,
  'token': 650,
  'token_str': ' small',
  'sequence': 'The Milky Way is a small galaxy.'}]
```

متن کو ٹوکنائز کریں اور `input_ids` کو PyTorch ٹینسر کے طور پر واپس کریں۔ آپ کو `<mask>` ٹوکن کی پوزیشن بھی بتانے کی ضرورت ہوگی:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer(text, return_tensors="pt")
>>> mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
```

اپنے ان پٹس کو ماڈل پر منتقل کریں اور ماسک شدہ ٹوکن کا `logits` واپس کریں:

```py
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("username/my_awesome_eli5_mlm_model")
>>> logits = model(**inputs).logits
>>> mask_token_logits = logits[0, mask_token_index, :]
```

پھر سب سے زیادہ امکان کے ساتھ تین نقاب پوش ٹوکن واپس کریں اور انہیں پرنٹ کریں:

```py
>>> top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0].tolist()

>>> for token in top_3_tokens:
...     print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))
The Milky Way is a spiral galaxy.
The Milky Way is a massive galaxy.
The Milky Way is a small galaxy.
```
