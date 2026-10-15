<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# متعدد انتخاب

[[open-in-colab]]

ایک سے زیادہ انتخاب کا کام سوال جواب دینے کی طرح ہوتا ہے، سوائے اس کے کہ امیدواروں کے متعدد جوابات سیاق و سباق کے ساتھ فراہم کیے جاتے ہیں اور ماڈل کو صحیح جواب منتخب کرنے کی تربیت دی جاتی ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [BERT](https://huggingface.co/google-bert/bert-base-uncased) `regular` کنفیگریشن پر [SWAG](https://huggingface.co/datasets/allenai/swag) ڈیٹاسیٹ کے متعدد اختیارات اور کچھ سیاق و سباق کے پیش نظر بہترین جواب کو منتخب کریں۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## SWAG ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹا سیٹس لائبریری سے SWAG ڈیٹاسیٹ کی `regular` کنفیگریشن لوڈ کر کے شروع کریں:

```py
>>> from datasets import load_dataset

>>> swag = load_dataset("allenai/swag", "regular")
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> swag["train"][0]
{'ending0': 'passes by walking down the street playing their instruments.',
 'ending1': 'has heard approaching them.',
 'ending2': "arrives and they're outside dancing and asleep.",
 'ending3': 'turns the lead singer watches the performance.',
 'fold-ind': '3416',
 'gold-source': 'gold',
 'label': 0,
 'sent1': 'Members of the procession walk down the street holding small horn brass instruments.',
 'sent2': 'A drum line',
 'startphrase': 'Members of the procession walk down the street holding small horn brass instruments. A drum line',
 'video-id': 'anetv_jkn6uvmqwh4'}
```

اگرچہ ایسا لگتا ہے کہ یہاں بہت سارے فیلڈز ہیں، یہ حقیقت میں بہت سیدھا ہے:

- `sent1` اور `sent2`: یہ فیلڈز دکھاتے ہیں کہ جملہ کیسے شروع ہوتا ہے، اور اگر آپ دونوں کو ایک ساتھ رکھتے ہیں، تو آپ کو `startphrase` فیلڈ ملے گی۔
- `ending`: ایک ممکنہ اختتام تجویز کرتا ہے کہ ایک جملہ کیسے ختم ہوسکتا ہے، لیکن ان میں سے صرف ایک درست ہے۔
- `label`: درست جملے کے اختتام کی نشاندہی کرتا ہے۔

## پری پروسیس

اگلا مرحلہ BERT ٹوکنائزر لوڈ کرنا ہے تاکہ جملے کے شروع ہونے اور چار ممکنہ اختتام پر کارروائی کی جا سکے۔

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
```

پری پروسیسنگ فنکشن جس کو آپ بنانا چاہتے ہیں اس کی ضرورت ہے:

1. `sent1` فیلڈ کی چار کاپیاں بنائیں اور ان میں سے ہر ایک کو `sent2` کے ساتھ جوڑ کر دوبارہ بنائیں کہ جملہ کیسے شروع ہوتا ہے۔
2. `sent2` کو چار ممکنہ جملے کے اختتام میں سے ہر ایک کے ساتھ جوڑیں۔
3. ان دونوں فہرستوں کو فلیٹ کریں تاکہ آپ ان کو ٹوکنائز کر سکیں، اور پھر ان کو بعد میں فلیٹ کریں تاکہ ہر مثال میں ایک متعلقہ `input_ids`، `attention_mask`، اور `labels` فیلڈ ہو۔

```py
>>> ending_names = ["ending0", "ending1", "ending2", "ending3"]


>>> def preprocess_function(examples):
...     first_sentences = [[context] * 4 for context in examples["sent1"]]
...     question_headers = examples["sent2"]
...     second_sentences = [
...         [f"{header} {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)
...     ]

...     first_sentences = sum(first_sentences, [])
...     second_sentences = sum(second_sentences, [])

...     tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
...     return {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] طریقہ استعمال کریں۔ آپ ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کر کے `map` فنکشن کو تیز کر سکتے ہیں:

```py
>>> tokenized_swag = swag.map(preprocess_function, batched=True)
```

مثالوں کا ایک بیچ بنانے کے لیے، مجموعہ کے دوران پورے ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے، جملے کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔ [`DataCollatorForMultipleChoice`] تمام ماڈل ان پٹس کو چپٹا کرتا ہے، پیڈنگ لگاتا ہے، اور پھر نتائج کو کھولتا ہے۔

```py
>>> from transformers import DataCollatorForMultipleChoice
>>> collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) میٹرک لوڈ کریں (میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے طریقے کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں):

```py
>>> import evaluate

>>> accuracy = evaluate.load("accuracy")
```

پھر ایک فنکشن بنائیں جو درستگی کا حساب لگانے کے لیے آپ کی پیشین گوئیوں اور لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے:

```py
>>> import numpy as np


>>> def compute_metrics(eval_pred):
...     predictions, labels = eval_pred
...     predictions = np.argmax(predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=labels)
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! BERT کو [`AutoModelForMultipleChoice`] کے ساتھ لوڈ کریں:

```py
>>> from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer

>>> model = AutoModelForMultipleChoice.from_pretrained("google-bert/bert-base-uncased")
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] درستگی کا اندازہ کرے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_swag_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     learning_rate=5e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_swag["train"],
...     eval_dataset=tokenized_swag["validation"],
...     processing_class=tokenizer,
...     data_collator=collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

ایک سے زیادہ انتخاب کے لیے ماڈل کو کیسے ٹھیک کیا جائے اس کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/multiple_choice.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

کچھ متن اور دو امیدواروں کے جوابات کے ساتھ آئیں:

```py
>>> prompt = "France has a bread law, Le Décret Pain, with strict rules on what is allowed in a traditional baguette."
>>> candidate1 = "The law does not apply to croissants and brioche."
>>> candidate2 = "The law applies to baguettes."
```

ہر پرامپٹ اور امیدوار جوابی جوڑے کو ٹوکنائز کریں اور PyTorch ٹینسر واپس کریں۔ آپ کو کچھ `labels` بھی بنانا چاہیے:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_swag_model")
>>> inputs = tokenizer([[prompt, candidate1], [prompt, candidate2]], return_tensors="pt", padding=True)
>>> labels = torch.tensor(0).unsqueeze(0)
```

اپنے ان پٹس اور لیبلز کو ماڈل پر منتقل کریں اور `logits` واپس کریں:

```py
>>> from transformers import AutoModelForMultipleChoice

>>> model = AutoModelForMultipleChoice.from_pretrained("username/my_awesome_swag_model")
>>> outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
>>> logits = outputs.logits
```

سب سے زیادہ امکان کے ساتھ کلاس حاصل کریں:

```py
>>> predicted_class = logits.argmax().item()
>>> predicted_class
0
```
