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

# سوال جواب

[[open-in-colab]]

<Youtube id="ajPx5LwJD-I"/>

سوال کا جواب دینے والے کام ایک سوال کا جواب دیتے ہیں۔ اگر آپ نے کبھی کسی ورچوئل اسسٹنٹ جیسے Alexa، Siri یا Google سے پوچھا ہے کہ موسم کیا ہے، تو آپ نے پہلے ماڈل جواب دینے والا سوال استعمال کیا ہے۔ سوالوں کے جواب دینے والے کاموں کی دو عام قسمیں ہیں:

- Extractive: دیئے گئے سیاق و سباق سے جواب نکالیں۔
- خلاصہ: سیاق و سباق سے ایک جواب تیار کریں جو سوال کا صحیح جواب دیتا ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased) استخراجی سوالوں کے جوابات کے لیے [SQuAD](https://huggingface.co/datasets/rajpurkar/squad) ڈیٹا سیٹ پر۔
2. انفرنس کے لیے اپنا فائن ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/question-answering) کو چیک کرنے کی تجویز کرتے ہیں۔

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

## Squad ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے SQuAD ڈیٹاسیٹ کا ایک چھوٹا سب سیٹ لوڈ کرکے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from datasets import load_dataset

>>> squad = load_dataset("rajpurkar/squad", split="train[:5000]")
```

ڈیٹا سیٹ کے `train` کو ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```py
>>> squad = squad.train_test_split(test_size=0.2)
```

پھر ایک مثال پر ایک نظر ڈالیں:

```py
>>> squad["train"][0]
{'answers': {'answer_start': [515], 'text': ['Saint Bernadette Soubirous']},
 'context': 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.',
 'id': '5733be284776f41900661182',
 'question': 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?',
 'title': 'University_of_Notre_Dame'
}
```

یہاں کئی اہم شعبے ہیں:

- `answers`: جوابی ٹوکن کا ابتدائی مقام اور جوابی متن۔
- `context`: پس منظر کی معلومات جس سے ماڈل کو جواب نکالنے کی ضرورت ہے۔
- `question`: سوال کا جواب ماڈل کو دینا چاہیے۔

## پری پروسیس

<Youtube id="qgaM0weJHpA"/>

اگلا مرحلہ `question` اور `context` فیلڈز پر کارروائی کرنے کے لیے DistilBERT ٹوکنائزر لوڈ کرنا ہے:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
```

سوالوں کے جواب دینے والے کاموں کے لیے خاص طور پر کچھ پری پروسیسنگ اقدامات ہیں جن سے آپ کو آگاہ ہونا چاہیے:

1. ڈیٹا سیٹ میں کچھ مثالیں بہت لمبی `context` ہو سکتی ہیں جو ماڈل کی زیادہ سے زیادہ ان پٹ طوالت سے زیادہ ہوتی ہیں۔ طویل ترتیب سے نمٹنے کے لیے، `truncation="only_second"` سیٹ کر کے صرف `context` کو چھوٹا کریں۔
2. اس کے بعد، ترتیب دے کر اصل `context` کے جواب کے آغاز اور اختتامی مقامات کا نقشہ بنائیں
`return_offset_mapping=True`۔
3. ہاتھ میں میپنگ کے ساتھ، اب آپ جواب کے آغاز اور اختتامی ٹوکن تلاش کر سکتے ہیں۔ [`~tokenizers.Encoding.sequence_ids`] طریقہ استعمال کریں۔
معلوم کریں کہ آفسیٹ کا کون سا حصہ `question` سے مطابقت رکھتا ہے اور کون سا `context` سے مماثل ہے۔

یہاں یہ ہے کہ آپ `answer` کے آغاز اور اختتامی ٹوکن کو `context` میں تراشنے اور نقشہ بنانے کے لیے ایک فنکشن کیسے بنا سکتے ہیں:

```py
>>> def preprocess_function(examples):
...     questions = [q.strip() for q in examples["question"]]
...     inputs = tokenizer(
...         questions,
...         examples["context"],
...         max_length=384,
...         truncation="only_second",
...         return_offsets_mapping=True,
...         padding="max_length",
...     )

...     offset_mapping = inputs.pop("offset_mapping")
...     answers = examples["answers"]
...     start_positions = []
...     end_positions = []

...     for i, offset in enumerate(offset_mapping):
...         answer = answers[i]
...         start_char = answer["answer_start"][0]
...         end_char = answer["answer_start"][0] + len(answer["text"][0])
...         sequence_ids = inputs.sequence_ids(i)

...         # Find the start and end of the context
...         idx = 0
...         while sequence_ids[idx] != 1:
...             idx += 1
...         context_start = idx
...         while sequence_ids[idx] == 1:
...             idx += 1
...         context_end = idx - 1

...         # If the answer is not fully inside the context, label it (0, 0)
...         if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
...             start_positions.append(0)
...             end_positions.append(0)
...         else:
...             # Otherwise it's the start and end token positions
...             idx = context_start
...             while idx <= context_end and offset[idx][0] <= start_char:
...                 idx += 1
...             start_positions.append(idx - 1)

...             idx = context_end
...             while idx >= context_start and offset[idx][1] >= end_char:
...                 idx -= 1
...             end_positions.append(idx + 1)

...     inputs["start_positions"] = start_positions
...     inputs["end_positions"] = end_positions
...     return inputs
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] فنکشن استعمال کریں۔ آپ ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پروسیس کرنے کے لیے `batched=True` سیٹ کر کے `map` فنکشن کو تیز کر سکتے ہیں۔ ایسے کالموں کو ہٹا دیں جن کی آپ کو ضرورت نہیں ہے:

```py
>>> tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
```

اب [`DefaultDataCollator`] کا استعمال کرتے ہوئے مثالوں کا ایک بیچ بنائیں۔ 🤗 Transformers میں دیگر ڈیٹا کولیٹرز کے برعکس، [`DefaultDataCollator`] کوئی اضافی پری پروسیسنگ جیسے پیڈنگ کا اطلاق نہیں کرتا ہے۔

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! [`AutoModelForQuestionAnswering`] کے ساتھ DistilBERT لوڈ کریں:

```py
>>> from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

>>> model = AutoModelForQuestionAnswering.from_pretrained("distilbert/distilbert-base-uncased")
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، اور ڈیٹا کولیٹر کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. اپنے ماڈل کو فائن ٹیون کرنے کے لیے [`~Trainer.train`] پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_qa_model",
...     eval_strategy="epoch",
...     learning_rate=2e-5,
...     per_device_train_batch_size=16,
...     per_device_eval_batch_size=16,
...     num_train_epochs=3,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=tokenized_squad["train"],
...     eval_dataset=tokenized_squad["test"],
...     processing_class=tokenizer,
...     data_collator=data_collator,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

سوال کے جواب کے لیے ماڈل کو کیسے ٹھیک کیا جائے اس کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/question_answering.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## اندازہ لگانا

سوال کے جواب کی تشخیص کے لیے کافی مقدار میں پوسٹ پروسیسنگ کی ضرورت ہوتی ہے۔ اپنا زیادہ وقت ضائع کرنے سے بچنے کے لیے، یہ گائیڈ تشخیصی مرحلے کو چھوڑ دیتا ہے۔ [`Trainer`] اب بھی تربیت کے دوران تشخیصی نقصان کا حساب لگاتا ہے تاکہ آپ اپنی ماڈل کی کارکردگی کے بارے میں مکمل طور پر اندھیرے میں نہ ہوں۔

اگر آپ کے پاس زیادہ وقت ہے اور آپ اس بات میں دلچسپی رکھتے ہیں کہ سوال کے جواب کے لیے اپنے ماڈل کا اندازہ کیسے لگایا جائے، 🤗 Hugging Face کورس سے [Question answering](https://huggingface.co/course/chapter7/7?fw=pt#post-processing) باب پر ایک نظر ڈالیں!

## انفرنس

بہت اچھا، اب جب کہ آپ نے ماڈل کو ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

ایک سوال اور کچھ سیاق و سباق کے ساتھ آئیں جس کی آپ ماڈل پیش گوئی کرنا چاہتے ہیں:

```py
>>> question = "How many programming languages does BLOOM support?"
>>> context = "BLOOM has 176 billion parameters and can generate text in 46 languages natural languages and 13 programming languages."
```

متن کو ٹوکنائز کریں اور PyTorch ٹینسر واپس کریں:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("my_awesome_qa_model")
>>> inputs = tokenizer(question, context, return_tensors="pt")
```

اپنے ان پٹس کو ماڈل پر دیں اور `logits` واپس کریں:

```py
>>> import torch
>>> from transformers import AutoModelForQuestionAnswering

>>> model = AutoModelForQuestionAnswering.from_pretrained("my_awesome_qa_model")
>>> with torch.no_grad():
...     outputs = model(**inputs)
```

آغاز اور اختتامی پوزیشنوں کے لیے ماڈل آؤٹ پٹ سے سب سے زیادہ امکان حاصل کریں:

```py
>>> answer_start_index = outputs.start_logits.argmax()
>>> answer_end_index = outputs.end_logits.argmax()
```

جواب حاصل کرنے کے لیے پیش گوئی شدہ ٹوکنز کو ڈی کوڈ کریں:

```py
>>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
>>> tokenizer.decode(predict_answer_tokens)
'176 billion parameters and can generate text in 46 languages natural languages and 13'
```
