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

# ٹوکن کی درجہ بندی

[[open-in-colab]]

<Youtube id="wVHdVlPScxA"/>

ٹوکن کی درجہ بندی ایک جملے میں انفرادی ٹوکن کو ایک لیبل تفویض کرتی ہے۔ ٹوکن کی درجہ بندی کے سب سے عام کاموں میں سے ایک نام کی ہستی کی شناخت (NER) ہے۔ NER ہر ایک ہستی کے لیے ایک جملے میں ایک لیبل تلاش کرنے کی کوشش کرتا ہے، جیسے کہ ایک شخص، مقام، یا تنظیم۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) [WNUT 17](https://huggingface.co/datasets/flaitenberger/wnut_17) ڈیٹاسیٹ پر نئی ہستیوں کا پتہ لگانے کے لیے۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس اس ٹاسک کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/token-classification) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate seqeval
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## WNUT 17 ڈیٹاسیٹ لوڈ کریں۔

WNUT 17 ڈیٹاسیٹ کو 🤗 ڈیٹاسیٹس لائبریری سے لوڈ کرکے شروع کریں:

```py
>>> from datasets import load_dataset

>>> wnut = load_dataset("flaitenberger/wnut_17")
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> wnut["train"][0]
{'id': '0',
 'ner_tags': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 8, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
 'tokens': ['@paulwalk', 'It', "'s", 'the', 'view', 'from', 'where', 'I', "'m", 'living', 'for', 'two', 'weeks', '.', 'Empire', 'State', 'Building', '=', 'ESB', '.', 'Pretty', 'bad', 'storm', 'here', 'last', 'evening', '.']
}
```

`ner_tags` میں ہر نمبر ایک ہستی کی نمائندگی کرتا ہے۔ نمبروں کو ان کے لیبل کے ناموں میں تبدیل کریں تاکہ معلوم ہو سکے کہ ادارے کیا ہیں:

```py
>>> label_list = wnut["train"].features[f"ner_tags"].feature.names
>>> label_list
[
    "O",
    "B-corporation",
    "I-corporation",
    "B-creative-work",
    "I-creative-work",
    "B-group",
    "I-group",
    "B-location",
    "I-location",
    "B-person",
    "I-person",
    "B-product",
    "I-product",
]
```

ہر ایک `ner_tag` کا سابقہ ​​لگانے والا خط ہستی کی ٹوکن پوزیشن کی نشاندہی کرتا ہے:

- `B-` کسی ہستی کے آغاز کی نشاندہی کرتا ہے۔
- `I-` اشارہ کرتا ہے کہ ایک ٹوکن ایک ہی ہستی کے اندر موجود ہے (مثال کے طور پر، `State` ٹوکن کسی ہستی کا حصہ ہے جیسے
`Empire State Building`)۔
- `0` اشارہ کرتا ہے کہ ٹوکن کسی بھی ہستی سے مطابقت نہیں رکھتا۔

## پری پروسیس

<Youtube id="iY2AZYdZAr0"/>

اگلا مرحلہ ایک DistilBERT ٹوکنائزر لوڈ کرنا ہے `tokens` فیلڈ کو پری پروسیس کرنے کے لیے:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

جیسا کہ آپ نے اوپر کی مثال `tokens` فیلڈ میں دیکھا، ایسا لگتا ہے کہ ان پٹ کو پہلے ہی ٹوکنائز کر دیا گیا ہے۔ لیکن اصل میں ان پٹ کو ابھی تک ٹوکنائز نہیں کیا گیا ہے اور آپ کو الفاظ کو ذیلی الفاظ میں ٹوکنائز کرنے کے لیے `is_split_into_words=True` سیٹ کرنے کی ضرورت ہوگی۔ مثال کے طور پر:

```py
>>> example = wnut["train"][0]
>>> tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
>>> tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
>>> tokens
['[CLS]', '@', 'paul', '##walk', 'it', "'", 's', 'the', 'view', 'from', 'where', 'i', "'", 'm', 'living', 'for', 'two', 'weeks', '.', 'empire', 'state', 'building', '=', 'es', '##b', '.', 'pretty', 'bad', 'storm', 'here', 'last', 'evening', '.', '[SEP]']
```

تاہم، اس میں کچھ خاص ٹوکنز `[CLS]` اور `[SEP]` شامل ہوتے ہیں اور ذیلی لفظ ٹوکنائزیشن ان پٹ اور لیبلز کے درمیان مماثلت پیدا کرتی ہے۔ کسی ایک لیبل سے مماثل ایک لفظ اب دو ذیلی الفاظ میں تقسیم ہو سکتا ہے۔ آپ کو ٹوکنز اور لیبلز کو دوبارہ ترتیب دینے کی ضرورت ہوگی بذریعہ:

1. [`~BatchEncoding#word_ids`] طریقہ کے ساتھ تمام ٹوکنز کو ان کے متعلقہ لفظ سے میپ کرنا۔
2. لیبل تفویض کرنا`-100`خصوصی ٹوکنز کے لیے`[CLS]`اور`[SEP]`لہذا وہ نظر انداز کر رہے ہیںPyTorchنقصان کا فنکشن (دیکھیں [CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)).
3. صرف دیئے گئے لفظ کے پہلے ٹوکن پر لیبل لگانا۔ اسی لفظ سے دوسرے سب ٹوکنز کو `-100` تفویض کریں۔

یہاں یہ ہے کہ آپ ٹوکنز اور لیبلز کو دوبارہ ترتیب دینے کے لیے ایک فنکشن کیسے بنا سکتے ہیں، اور ترتیب کو چھوٹا کر سکتے ہیں تاکہ DistilBERT کی زیادہ سے زیادہ ان پٹ لمبائی سے زیادہ نہ ہو:

```py
>>> def tokenize_and_align_labels(examples):
...     tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

...     labels = []
...     for i, label in enumerate(examples[f"ner_tags"]):
...         word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
...         previous_word_idx = None
...         label_ids = []
...         for word_idx in word_ids:  # Set the special tokens to -100.
...             if word_idx is None:
...                 label_ids.append(-100)
...             elif word_idx != previous_word_idx:  # Only label the first token of a given word.
...                 label_ids.append(label[word_idx])
...             else:
...                 label_ids.append(-100)
...             previous_word_idx = word_idx
...         labels.append(label_ids)

...     tokenized_inputs["labels"] = labels
...     return tokenized_inputs
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] فنکشن استعمال کریں۔ آپ ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کر کے `map` فنکشن کو تیز کر سکتے ہیں:

```py
>>> tokenized_wnut = wnut.map(tokenize_and_align_labels, batched=True)
```

اب [`DataCollatorWithPadding`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ مجموعہ کے دوران تمام ڈیٹاسیٹ کو زیادہ سے زیادہ لمبائی میں پیڈ کرنے کے بجائے جملوں کو *متحرک طور پر پیڈ* کرنا زیادہ کارآمد ہے۔

```py
>>> from transformers import DataCollatorForTokenClassification

>>> data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [seqeval](https://huggingface.co/spaces/evaluate-metric/seqeval) فریم ورک کو لوڈ کریں (میٹرک کو لوڈ کرنے اور اس کی گنتی کرنے کے طریقے کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں)۔ Seqeval دراصل کئی اسکور تیار کرتا ہے: درستگی، یاد، F1، اور درستگی۔

```py
>>> import evaluate

>>> seqeval = evaluate.load("seqeval")
```

پہلے NER لیبلز حاصل کریں، اور پھر ایک ایسا فنکشن بنائیں جو آپ کی صحیح پیشین گوئیوں اور حقیقی لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے تاکہ اسکورز کا حساب لگایا جا سکے۔

```py
>>> import numpy as np

>>> labels = [label_list[i] for i in example[f"ner_tags"]]


>>> def compute_metrics(p):
...     predictions, labels = p
...     predictions = np.argmax(predictions, axis=2)

...     true_predictions = [
...         [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]
...     true_labels = [
...         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
...         for prediction, label in zip(predictions, labels)
...     ]

...     results = seqeval.compute(predictions=true_predictions, references=true_labels)
...     return {
...         "precision": results["overall_precision"],
...         "recall": results["overall_recall"],
...         "f1": results["overall_f1"],
...         "accuracy": results["overall_accuracy"],
...     }
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

اپنے ماڈل کی تربیت شروع کرنے سے پہلے، `id2label` اور `label2id` کے ساتھ ان کے لیبل پر متوقع ids کا نقشہ بنائیں:

```py
>>> id2label = {
...     0: "O",
...     1: "B-corporation",
...     2: "I-corporation",
...     3: "B-creative-work",
...     4: "I-creative-work",
...     5: "B-group",
...     6: "I-group",
...     7: "B-location",
...     8: "I-location",
...     9: "B-person",
...     10: "I-person",
...     11: "B-product",
...     12: "I-product",
... }
>>> label2id = {
...     "O": 0,
...     "B-corporation": 1,
...     "I-corporation": 2,
...     "B-creative-work": 3,
...     "I-creative-work": 4,
...     "B-group": 5,
...     "I-group": 6,
...     "B-location": 7,
...     "I-location": 8,
...     "B-person": 9,
...     "I-person": 10,
...     "B-product": 11,
...     "I-product": 12,
... }
```

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! DistilBERT کو [`AutoModelForTokenClassification`] کے ساتھ متوقع لیبلز کی تعداد اور لیبل میپنگ کے ساتھ لوڈ کریں:

```py
>>> from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

>>> model = AutoModelForTokenClassification.from_pretrained(
...     "distilbert/distilbert-base-uncased", num_labels=13, id2label=id2label, label2id=label2id
... )
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] seqeval اسکورز کا جائزہ لے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_wnut_model",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=2,
...     weight_decay=0.01,
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     load_best_model_at_end=True,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_wnut["train"],
...     eval_dataset=tokenized_wnut["test"],
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

ٹوکن کی درجہ بندی کے لیے ماڈل کو ٹھیک کرنے کے طریقے کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

کچھ متن پکڑو جس پر آپ انفرنس چلانا چاہتے ہیں:

```py
>>> text = "The Golden State Warriors are an American professional basketball team based in San Francisco."
```

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ NER کے لیے ایک `pipeline` فوری بنائیں، اور اپنا متن اس تک پہنچائیں:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("ner", model="stevhliu/my_awesome_wnut_model")
>>> classifier(text)
[{'entity': 'B-location',
  'score': 0.42658573,
  'index': 2,
  'word': 'golden',
  'start': 4,
  'end': 10},
 {'entity': 'I-location',
  'score': 0.35856336,
  'index': 3,
  'word': 'state',
  'start': 11,
  'end': 16},
 {'entity': 'B-group',
  'score': 0.3064001,
  'index': 4,
  'word': 'warriors',
  'start': 17,
  'end': 25},
 {'entity': 'B-location',
  'score': 0.65523505,
  'index': 13,
  'word': 'san',
  'start': 80,
  'end': 83},
 {'entity': 'B-location',
  'score': 0.4668663,
  'index': 14,
  'word': 'francisco',
  'start': 84,
  'end': 93}]
```

اگر آپ چاہیں تو آپ `pipeline` کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں:

متن کو ٹوکنائز کریں اور PyTorch ٹینسر واپس کریں:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> inputs = tokenizer(text, return_tensors="pt")
```

اپنے ان پٹس کو ماڈل پر دیں اور `logits` واپس کریں:

```py
>>> from transformers import AutoModelForTokenClassification

>>> model = AutoModelForTokenClassification.from_pretrained("stevhliu/my_awesome_wnut_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

سب سے زیادہ امکان کے ساتھ کلاس حاصل کریں، اور اسے ٹیکسٹ لیبل میں تبدیل کرنے کے لیے ماڈل کی `id2label` میپنگ کا استعمال کریں:

```py
>>> predictions = torch.argmax(logits, dim=2)
>>> predicted_token_class = [model.config.id2label[t.item()] for t in predictions[0]]
>>> predicted_token_class
['O',
 'O',
 'B-location',
 'I-location',
 'B-group',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'O',
 'B-location',
 'B-location',
 'O',
 'O']
```
