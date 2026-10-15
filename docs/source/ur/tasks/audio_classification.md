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

# آڈیو کی درجہ بندی

[[open-in-colab]]

<Youtube id="KWwzcmG98Ds"/>

آڈیو کی درجہ بندی - بالکل متن کی طرح - ایک کلاس لیبل کو ان پٹ ڈیٹا سے آؤٹ پٹ کے طور پر تفویض کرتا ہے۔ فرق صرف ٹیکسٹ ان پٹ کے بجائے ہے، آپ کے پاس خام آڈیو ویوفارمز ہیں۔ آڈیو درجہ بندی کی کچھ عملی ایپلی کیشنز میں بولنے والے کے ارادے، زبان کی درجہ بندی، اور یہاں تک کہ جانوروں کی انواع کو ان کی آوازوں سے شناخت کرنا شامل ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) ڈیٹا سیٹ پر اسپیکر کے ارادے کی درجہ بندی کرنے کے لیے۔
2. انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/audio-classification) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate soundfile librosa torchcodec
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## MINDS-14 ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے MINDS-14 ڈیٹاسیٹ لوڈ کرکے شروع کریں:

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train")
```

ڈیٹا سیٹ کے `train` کو ایک چھوٹی ٹرین میں تقسیم کریں اور [`~datasets.Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

پھر ڈیٹاسیٹ پر ایک نظر ڈالیں:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 450
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 113
    })
})
```

جبکہ ڈیٹاسیٹ میں بہت ساری مفید معلومات ہیں، جیسے `lang_id` اور `english_transcription`، آپ اس گائیڈ میں `audio` اور `intent_class` پر توجہ مرکوز کریں گے۔ دوسرے کالموں کو [`~datasets.Dataset.remove_columns`] طریقہ سے ہٹائیں:

```py
>>> minds = minds.remove_columns(["path", "transcription", "english_transcription", "lang_id"])
```

یہاں ایک مثال ہے:

```py
>>> minds["train"][0]
{'audio': {'array': array([ 0.        ,  0.        ,  0.        , ..., -0.00048828,
         -0.00024414, -0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 8000},
 'intent_class': 2}
```

دو میدان ہیں:

- `audio`: اسپیچ سگنل کا ایک جہتی `array` جسے آڈیو فائل کو لوڈ کرنے اور دوبارہ نمونہ کرنے کے لیے کال کرنا ضروری ہے۔
- `intent_class`: اسپیکر کے ارادے کی کلاس آئی ڈی کی نمائندگی کرتا ہے۔

ماڈل کے لیے لیبل آئی ڈی سے لیبل کا نام حاصل کرنا آسان بنانے کے لیے، ایک لغت بنائیں جو لیبل کے نام کو ایک عدد میں نقشے اور اس کے برعکس:

```py
>>> labels = minds["train"].features["intent_class"].names
>>> label2id, id2label = dict(), dict()
>>> for i, label in enumerate(labels):
...     label2id[label] = str(i)
...     id2label[str(i)] = label
```

اب آپ لیبل آئی ڈی کو لیبل کے نام میں تبدیل کر سکتے ہیں:

```py
>>> id2label[str(2)]
'app_error'
```

## پری پروسیس

اگلا مرحلہ آڈیو سگنل پر کارروائی کرنے کے لیے Wav2Vec2 فیچر ایکسٹریکٹر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
```

MINDS-14 ڈیٹاسیٹ کا نمونہ لینے کی شرح 8kHz ہے (آپ یہ معلومات اس کے [ڈیٹا سیٹ کارڈ میں تلاش کر سکتے ہیں](https://huggingface.co/datasets/PolyAI/minds14))، جس کا مطلب ہے کہ آپ کو استعمال کرنے کے لیے ڈیٹاسیٹ کو 16kHz پر دوبارہ نمونہ کرنے کی ضرورت ہوگی۔پہلے سے تربیت یافتہWav2Vec2ماڈل:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([ 2.2098757e-05,  4.6582241e-05, -2.2803260e-05, ...,
         -2.8419291e-04, -2.3305941e-04, -1.1425107e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602b9a5fbb1e6d0fbce91f52.wav',
  'sampling_rate': 16000},
 'intent_class': 2}
```

اب ایک پری پروسیسنگ فنکشن بنائیں جو:

1. `audio` کالم کو لوڈ کرنے کے لیے کال کرتا ہے، اور اگر ضروری ہو تو، آڈیو فائل کو دوبارہ نمونہ کریں۔
2. چیک کرتا ہے کہ آیا آڈیو فائل کے نمونے لینے کی شرح آڈیو ڈیٹا کے نمونے لینے کی شرح سے ملتی ہے جس کے ساتھ ماڈل پہلے سے تربیت یافتہ تھا۔ آپ یہ معلومات Wav2Vec2 [ماڈل card](https://huggingface.co/facebook/wav2vec2-base) میں حاصل کر سکتے ہیں۔
3. لمبے لمبے ان پٹس کو تراشے بغیر بیچنے کے لیے زیادہ سے زیادہ ان پٹ کی لمبائی سیٹ کریں۔

```py
>>> def preprocess_function(examples):
...     audio_arrays = [x["array"] for x in examples["audio"]]
...     inputs = feature_extractor(
...         audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=16000, truncation=True
...     )
...     return inputs
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] فنکشن استعمال کریں۔ آپ `map` سیٹ کر کے `batched=True` ڈیٹا سیٹ کے متعدد عناصر کو ایک ساتھ پراسیس کر سکتے ہیں۔ غیر ضروری کالم ہٹائیں اور `intent_class` کا نام `label` میں تبدیل کریں، جیسا کہ ماڈل کی ضرورت ہے:

```py
>>> encoded_minds = minds.map(preprocess_function, remove_columns="audio", batched=True)
>>> encoded_minds = encoded_minds.rename_column("intent_class", "label")
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
...     predictions = np.argmax(eval_pred.predictions, axis=1)
...     return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

آپ ابھی اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! Wav2Vec2 کو [`AutoModelForAudioClassification`] کے ساتھ متوقع لیبلز کی تعداد اور لیبل میپنگ کے ساتھ لوڈ کریں:

```py
>>> from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer

>>> num_labels = len(id2label)
>>> model = AutoModelForAudioClassification.from_pretrained(
...     "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label
... )
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے، جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] درستگی کا اندازہ کرے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. [`~Trainer.train`] کو فائن ٹیون اپنے ماڈل پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_mind_model",
...     eval_strategy="epoch",
...     save_strategy="epoch",
...     learning_rate=3e-5,
...     per_device_train_batch_size=32,
...     gradient_accumulation_steps=4,
...     per_device_eval_batch_size=32,
...     num_train_epochs=10,
...     warmup_steps=0.1,
...     logging_steps=10,
...     load_best_model_at_end=True,
...     metric_for_best_model="accuracy",
...     push_to_hub=True,
...     report_to="trackio",
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=feature_extractor,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ ہر کوئی آپ کے ماڈل کو استعمال کر سکے:

```py
>>> trainer.push_to_hub()
```

<Tip>

آڈیو کی درجہ بندی کے لیے فائن ٹیون a ماڈل کو کیسے بنایا جائے اس کی مزید گہرائی سے مثال کے لیے، متعلقہ [PyTorch نوٹ بک](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/audio_classification.ipynb) پر ایک نظر ڈالیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ایک ماڈل ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

انفرنس کے لیے ایک آڈیو فائل لوڈ کریں۔ اگر ضروری ہو تو ماڈل کے نمونے لینے کی شرح سے ملنے کے لیے آڈیو فائل کے نمونے لینے کی شرح کو دوبارہ نمونہ بنانا یاد رکھیں۔

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", name="en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ آڈیو کی درجہ بندی کے لیے ایک `pipeline` فوری بنائیں، اور اپنی آڈیو فائل کو اس میں منتقل کریں:

```py
>>> from transformers import pipeline

>>> classifier = pipeline("audio-classification", model="stevhliu/my_awesome_minds_model")
>>> classifier(audio_file)
[
    {'score': 0.09766869246959686, 'label': 'cash_deposit'},
    {'score': 0.07998877018690109, 'label': 'app_error'},
    {'score': 0.0781070664525032, 'label': 'joint_account'},
    {'score': 0.07667109370231628, 'label': 'pay_bill'},
    {'score': 0.0755252093076706, 'label': 'balance'}
]
```

اگر آپ چاہیں تو آپ `pipeline` کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں:

آڈیو فائل کو پری پروسیس کرنے کے لیے فیچر ایکسٹریکٹر لوڈ کریں اور `input` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoFeatureExtractor

>>> feature_extractor = AutoFeatureExtractor.from_pretrained("stevhliu/my_awesome_minds_model")
>>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

اپنے ان پٹس کو ماڈل پر منتقل کریں اور لاگٹس واپس کریں:

```py
>>> from transformers import AutoModelForAudioClassification

>>> model = AutoModelForAudioClassification.from_pretrained("stevhliu/my_awesome_minds_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

سب سے زیادہ امکان کے ساتھ کلاس حاصل کریں، اور اسے لیبل میں تبدیل کرنے کے لیے ماڈل کی `id2label` میپنگ کا استعمال کریں:

```py
>>> import torch

>>> predicted_class_ids = torch.argmax(logits).item()
>>> predicted_label = model.config.id2label[predicted_class_ids]
>>> predicted_label
'cash_deposit'
```
