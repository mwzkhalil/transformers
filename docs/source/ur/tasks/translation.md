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

# ترجمہ

[[open-in-colab]]

<Youtube id="1JvfrvZgi6c"/>

ترجمہ متن کی ترتیب کو ایک زبان سے دوسری زبان میں تبدیل کرتا ہے۔ یہ متعدد کاموں میں سے ایک ہے جسے آپ ترتیب سے ترتیب کے مسئلے کے طور پر تشکیل دے سکتے ہیں، ایک ان پٹ سے کچھ آؤٹ پٹ واپس کرنے کے لیے ایک طاقتور فریم ورک، جیسے ترجمہ یا خلاصہ۔ ترجمے کے نظام کو عام طور پر مختلف زبانوں کے متنوں کے درمیان ترجمہ کے لیے استعمال کیا جاتا ہے، لیکن یہ تقریر یا متن سے تقریر یا تقریر سے متن کے درمیان کچھ امتزاج کے لیے بھی استعمال کیا جا سکتا ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. انگریزی متن کا فرانسیسی میں ترجمہ کرنے کے لیے [OPUS Books](https://huggingface.co/google-t5/t5-small) ڈیٹاسیٹ کے انگریزی-فرانسیسی سب سیٹ پر فائن ٹیون [T5](https://huggingface.co/datasets/Helsinki-NLP/opus_books)۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس اس ٹاسک کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/translation) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate sacrebleu
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## OPUS Books ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے [OPUS Books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) ڈیٹاسیٹ کے انگریزی-فرانسیسی سب سیٹ کو لوڈ کرکے شروع کریں:

```py
>>> from datasets import load_dataset

>>> books = load_dataset("Helsinki-NLP/opus_books", "en-fr")
```

ڈیٹاسیٹ کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ سے ٹیسٹ سیٹ کریں:

```py
>>> books = books["train"].train_test_split(test_size=0.2)
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> books["train"][0]
{'id': '90560',
 'translation': {'en': 'But this lofty plateau measured only a few fathoms, and soon we reentered Our Element.',
  'fr': 'Mais ce plateau élevé ne mesurait que quelques toises, et bientôt nous fûmes rentrés dans notre élément.'}}
```

`translation`: متن کا انگریزی اور فرانسیسی ترجمہ۔

## پری پروسیس

<Youtube id="XAR8jnZZuUs"/>

اگلا مرحلہ انگریزی-فرانسیسی زبان کے جوڑوں پر کارروائی کرنے کے لیے T5 ٹوکنائزر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoTokenizer

>>> checkpoint = "google-t5/t5-small"
>>> tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

پری پروسیسنگ فنکشن جس کو آپ بنانا چاہتے ہیں اس کی ضرورت ہے:

1. پرامپٹ کے ساتھ ان پٹ کو پریفکس کریں تاکہ T5 کو معلوم ہو کہ یہ ترجمہ کا کام ہے۔ کچھ ماڈلز ایک سے زیادہ NLP کاموں کے لیے مخصوص کاموں کے لیے اشارہ کرنے کی ضرورت ہوتی ہے۔
2. ہدف کی زبان (فرانسیسی) کو `text_target` پیرامیٹر میں سیٹ کریں تاکہ یہ یقینی بنایا جا سکے کہ ٹوکنائزر ہدف کے متن کو درست طریقے سے پروسیس کرتا ہے۔ اگر آپ `text_target` سیٹ نہیں کرتے ہیں تو ٹوکنائزر ہدف کے متن کو انگریزی کے طور پر پروسیس کرتا ہے۔
3. `max_length` پیرامیٹر کے ذریعے ترتیب دی گئی زیادہ سے زیادہ لمبائی سے زیادہ لمبا نہ ہونے کے لیے ترتیب کو چھوٹا کریں۔

```py
>>> source_lang = "en"
>>> target_lang = "fr"
>>> prefix = "translate English to French: "


>>> def preprocess_function(examples):
...     inputs = [prefix + example[source_lang] for example in examples["translation"]]
...     targets = [example[target_lang] for example in examples["translation"]]
...     model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
...     return model_inputs
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] طریقہ استعمال کریں۔ آپ ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کر کے `map` فنکشن کو تیز کر سکتے ہیں:

```py
>>> tokenized_books = books.map(preprocess_function, batched=True)
```

اب [`DataCollatorForSeq2Seq`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ مجموعہ کے دوران تمام ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے جملوں کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔

```py
>>> from transformers import DataCollatorForSeq2Seq

>>> data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [SacreBLEU](https://huggingface.co/spaces/evaluate-metric/sacrebleu) میٹرک لوڈ کریں (میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے طریقے کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں):

```py
>>> import evaluate

>>> metric = evaluate.load("sacrebleu")
```

پھر ایک فنکشن بنائیں جو SacreBLEU سکور کا حساب لگانے کے لیے آپ کی پیشین گوئیوں اور لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے:

```py
>>> import numpy as np


>>> def postprocess_text(preds, labels):
...     preds = [pred.strip() for pred in preds]
...     labels = [[label.strip()] for label in labels]

...     return preds, labels


>>> def compute_metrics(eval_preds):
...     preds, labels = eval_preds
...     if isinstance(preds, tuple):
...         preds = preds[0]
...     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

...     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
...     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

...     decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

...     result = metric.compute(predictions=decoded_preds, references=decoded_labels)
...     result = {"bleu": result["score"]}

...     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
...     result["gen_len"] = np.mean(prediction_lens)
...     result = {k: round(v, 4) for k, v in result.items()}
...     return result
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForSeq2SeqLM`] کے ساتھ T5 لوڈ کریں:

```py
>>> from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

>>> model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`Seq2SeqTrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] SacreBLEU میٹرک کی جانچ کرے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Seq2SeqTrainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="my_awesome_opus_books_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     weight_decay=0.01,
...     save_total_limit=3,
...     num_train_epochs=2,
...     predict_with_generate=True,
...     fp16=True, #change to bf16=True for XPU
...     push_to_hub=True,
... )

>>> trainer = Seq2SeqTrainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_books["train"],
...     eval_dataset=tokenized_books["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

ترجمے کے لیے ماڈل کو ٹھیک کرنے کے طریقے کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/translation.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

کچھ متن کے ساتھ آئیں جس کا آپ دوسری زبان میں ترجمہ کرنا چاہتے ہیں۔ T5 کے لیے، آپ جس کام پر کام کر رہے ہیں اس کے لحاظ سے آپ کو اپنے ان پٹ کا سابقہ ​​لگانے کی ضرورت ہے۔ انگریزی سے فرانسیسی میں ترجمہ کے لیے، آپ کو اپنے ان پٹ کا سابقہ ​​لگانا چاہیے جیسا کہ ذیل میں دکھایا گیا ہے:

```py
>>> text = "translate English to French: Legumes share resources with nitrogen-fixing bacteria."
```

متن کو ٹوکنائز کریں اور `input_ids` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_opus_books_model")
>>> inputs = tokenizer(text, return_tensors="pt").input_ids
```

ترجمہ بنانے کے لیے [`~generation.GenerationMixin.generate`] طریقہ استعمال کریں۔ متن بنانے کی مختلف حکمت عملیوں اور جنریشن کو کنٹرول کرنے کے پیرامیٹرز کے بارے میں مزید تفصیلات کے لیے، [Text Generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) API کو دیکھیں۔

```py
>>> from transformers import AutoModelForSeq2SeqLM

>>> model = AutoModelForSeq2SeqLM.from_pretrained("username/my_awesome_opus_books_model")
>>> outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=30, top_p=0.95)
```

تیار کردہ ٹوکن آئی ڈی کو واپس متن میں ڈی کوڈ کریں:

```py
>>> tokenizer.decode(outputs[0], skip_special_tokens=True)
'Les lignées partagent des ressources avec des bactéries enfixant l'azote.'
```
