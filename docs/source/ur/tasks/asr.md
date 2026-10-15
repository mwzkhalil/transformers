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

# خودکار تقریر کی شناخت

[[open-in-colab]]

<Youtube id="TksaY_FDgnk"/>

خودکار اسپیچ ریکگنیشن (ASR) اسپیچ سگنل کو ٹیکسٹ میں تبدیل کرتا ہے، آڈیو ان پٹس کی ترتیب کو ٹیکسٹ آؤٹ پٹس میں میپ کرتا ہے۔ Siri اور Alexa جیسے ورچوئل اسسٹنٹس ہر روز صارفین کی مدد کے لیے ASR ماڈلز استعمال کرتے ہیں، اور میٹنگوں کے دوران لائیو کیپشننگ اور نوٹ لینے جیسی بہت سی دوسری مفید صارف کا سامنا کرنے والی ایپلیکیشنز ہیں۔

یہ گائیڈ آپ کو دکھائے گا کہ کیسے:

1. فائن ٹیون [Wav2Vec2](https://huggingface.co/facebook/wav2vec2-base) [MINDS-14](https://huggingface.co/datasets/PolyAI/minds14) ڈیٹاسیٹ پر آڈیو کو متن میں نقل کرنے کے لیے۔
2. انفرنس کے لیے اپنا ٹھیک ٹیونڈ ماڈل استعمال کریں۔

<Tip>

تمام فن تعمیرات اور چیک پوائنٹس کو اس کام کے ساتھ ہم آہنگ دیکھنے کے لیے، ہم [task-page](https://huggingface.co/tasks/automatic-speech-recognition) کو چیک کرنے کی تجویز کرتے ہیں۔

</Tip>

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets evaluate jiwer soundfile librosa torchcodec
```

ہم آپ کو اپنے Hugging Face اکاؤنٹ میں لاگ ان کرنے کی ترغیب دیتے ہیں تاکہ آپ اپنا ماڈل اپ لوڈ اور کمیونٹی کے ساتھ شیئر کر سکیں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## MINDS-14 ڈیٹاسیٹ لوڈ کریں۔

🤗 ڈیٹاسیٹس لائبریری سے [MInDS-14](https://huggingface.co/datasets/PolyAI/minds14) ڈیٹاسیٹ کا ایک چھوٹا سب سیٹ لوڈ کرکے شروع کریں۔ یہ آپ کو تجربہ کرنے کا موقع فراہم کرے گا اور اس بات کو یقینی بنائے گا کہ مکمل ڈیٹا سیٹ پر زیادہ وقت کی تربیت گزارنے سے پہلے سب کچھ کام کر رہا ہے۔

```py
>>> from datasets import load_dataset, Audio

>>> minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
```

ڈیٹا سیٹ کے `train` کو ٹرین میں تقسیم کریں اور [`~Dataset.train_test_split`] طریقہ کے ساتھ ٹیسٹ سیٹ:

```py
>>> minds = minds.train_test_split(test_size=0.2)
```

پھر ڈیٹاسیٹ پر ایک نظر ڈالیں:

```py
>>> minds
DatasetDict({
    train: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 16
    })
    test: Dataset({
        features: ['path', 'audio', 'transcription', 'english_transcription', 'intent_class', 'lang_id'],
        num_rows: 4
    })
})
```

جبکہ ڈیٹاسیٹ میں بہت ساری مفید معلومات ہیں، جیسے `lang_id` اور `english_transcription`، یہ گائیڈ `audio` اور `transcription` پر فوکس کرتا ہے۔ دوسرے کالموں کو [`~datasets.Dataset.remove_columns`] طریقہ سے ہٹائیں:

```py
>>> minds = minds.remove_columns(["english_transcription", "intent_class", "lang_id"])
```

مثال کا دوبارہ جائزہ لیں:

```py
>>> minds["train"][0]
{'audio': {'array': array([-0.00024414,  0.        ,  0.        , ...,  0.00024414,
          0.00024414,  0.00024414], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 8000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

دو میدان ہیں:

- `audio`: اسپیچ سگنل کا ایک جہتی `array` جسے آڈیو فائل کو لوڈ کرنے اور دوبارہ نمونہ کرنے کے لیے کال کرنا ضروری ہے۔
- `transcription`: ہدف کا متن۔

## پری پروسیس

اگلا مرحلہ آڈیو سگنل پر کارروائی کرنے کے لیے Wav2Vec2 پروسیسر لوڈ کرنا ہے۔

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")
```

MINDS-14 ڈیٹاسیٹ کا نمونہ لینے کی شرح 8000Hz ہے (آپ یہ معلومات اس کے [ڈیٹا سیٹ کارڈ میں تلاش کر سکتے ہیں](https://huggingface.co/datasets/PolyAI/minds14))، جس کا مطلب ہے کہ آپ کو استعمال کرنے کے لیے ڈیٹاسیٹ کو 16000Hz پر دوبارہ نمونہ کرنے کی ضرورت ہوگی۔پہلے سے تربیت یافتہWav2Vec2ماڈل:

```py
>>> minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
>>> minds["train"][0]
{'audio': {'array': array([-2.38064706e-04, -1.58618059e-04, -5.43987835e-06, ...,
          2.78103951e-04,  2.38446111e-04,  1.18740834e-04], dtype=float32),
  'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
  'sampling_rate': 16000},
 'path': '/root/.cache/huggingface/datasets/downloads/extracted/f14948e0e84be638dd7943ac36518a4cf3324e8b7aa331c5ab11541518e9368c/en-US~APP_ERROR/602ba9e2963e11ccd901cd4f.wav',
 'transcription': "hi I'm trying to use the banking app on my phone and currently my checking and savings account balance is not refreshing"}
```

جیسا کہ آپ اوپر `transcription` میں دیکھ سکتے ہیں، متن میں بڑے اور چھوٹے حروف کا مرکب ہوتا ہے۔ Wav2Vec2 ٹوکنائزر صرف بڑے حروف پر تربیت یافتہ ہے لہذا آپ کو یہ یقینی بنانا ہوگا کہ متن ٹوکنائزر کے الفاظ سے میل کھاتا ہے:

```py
>>> def uppercase(example):
...     return {"transcription": example["transcription"].upper()}


>>> minds = minds.map(uppercase)
```

اب ایک پری پروسیسنگ فنکشن بنائیں جو:

1. آڈیو فائل کو لوڈ کرنے اور دوبارہ نمونہ بنانے کے لیے `audio` کالم کو کال کرتا ہے۔
2. آڈیو فائل سے `input_values` نکالتا ہے اور `transcription` کالم کو پروسیسر کے ساتھ ٹوکنائز کرتا ہے۔

```py
>>> def prepare_dataset(batch):
...     audio = batch["audio"]
...     batch = processor(audio["array"], sampling_rate=audio["sampling_rate"], text=batch["transcription"])
...     batch["input_length"] = len(batch["input_values"][0])
...     return batch
```

پورے ڈیٹاسیٹ پر پری پروسیسنگ فنکشن لاگو کرنے کے لیے، 🤗 ڈیٹا سیٹس [`~datasets.Dataset.map`] فنکشن استعمال کریں۔ آپ `num_proc` پیرامیٹر کے ساتھ عمل کی تعداد بڑھا کر `map` رفتار بڑھا سکتے ہیں۔ [`~datasets.Dataset.remove_columns`] طریقہ سے ان کالموں کو ہٹا دیں جن کی آپ کو ضرورت نہیں ہے:

```py
>>> encoded_minds = minds.map(prepare_dataset, remove_columns=minds.column_names["train"], num_proc=4)
```

🤗 Transformers میں ASR کے لیے ڈیٹا کولیٹر نہیں ہے، لہذا آپ کو مثالوں کا ایک بیچ بنانے کے لیے [`DataCollatorWithPadding`] کو اپنانا ہوگا۔ یہ متحرک طور پر آپ کے متن اور لیبلز کو اس کے بیچ میں سب سے لمبے عنصر کی لمبائی میں پیڈ کرے گا (پورے ڈیٹاسیٹ کے بجائے) تاکہ وہ یکساں لمبائی کے ہوں۔ اگرچہ `padding=True` ترتیب دے کر `tokenizer` فنکشن میں اپنے متن کو پیڈ کرنا ممکن ہے، لیکن متحرک پیڈنگ زیادہ موثر ہے۔

دوسرے ڈیٹا کولیٹرز کے برعکس، اس مخصوص ڈیٹا کولیٹر کو `input_values` اور `labels` پر پیڈنگ کا ایک مختلف طریقہ لاگو کرنے کی ضرورت ہے:

```py
>>> import torch

>>> from dataclasses import dataclass, field
>>> from typing import Any, Dict, List, Optional, Union


>>> @dataclass
... class DataCollatorCTCWithPadding:
...     processor: AutoProcessor
...     padding: Union[bool, str] = "longest"

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         # split inputs and labels since they have to be of different lengths and need
...         # different padding methods
...         input_features = [{"input_values": feature["input_values"][0]} for feature in features]
...         label_features = [{"input_ids": feature["labels"]} for feature in features]

...         batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")

...         labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

...         batch["labels"] = labels

...         return batch
```

اب اپنا `DataCollatorCTCWithPadding` فوری بنائیں:

```py
>>> data_collator = DataCollatorCTCWithPadding(processor=processor, padding="longest")
```

## اندازہ لگانا

ٹریننگ کے دوران میٹرک شامل کرنا اکثر آپ کی ماڈل کی کارکردگی کا جائزہ لینے میں مددگار ہوتا ہے۔ آپ 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) لائبریری کے ساتھ تشخیص کا طریقہ تیزی سے لوڈ کر سکتے ہیں۔ اس کام کے لیے، [لفظ کی خرابی کی شرح](https://huggingface.co/spaces/evaluate-metric/wer) (WER) میٹرک لوڈ کریں (لوڈنگ اور کمپیوٹنگ میٹرکس کے بارے میں مزید جاننے کے لیے 🤗 تشخیص [کوئیک ٹور](https://huggingface.co/docs/evaluate/a_quick_tour) دیکھیں):

```py
>>> import evaluate

>>> wer = evaluate.load("wer")
```

پھر ایک فنکشن بنائیں جو WER کا حساب لگانے کے لیے آپ کی پیشین گوئیوں اور لیبلز کو [`~evaluate.EvaluationModule.compute`] پر منتقل کرے:

```py
>>> import numpy as np


>>> def compute_metrics(pred):
...     pred_logits = pred.predictions
...     pred_ids = np.argmax(pred_logits, axis=-1)

...     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

...     pred_str = processor.batch_decode(pred_ids)
...     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

...     wer_score = wer.compute(predictions=pred_str, references=label_str)

...     return {"wer": wer_score}
```

آپ کا `compute_metrics` فنکشن اب جانے کے لیے تیار ہے، اور جب آپ اپنی ٹریننگ سیٹ اپ کریں گے تو آپ اس پر واپس آ جائیں گے۔

## ٹرین

<Tip>

اگر آپ ماڈل کو [`Trainer`] کے ساتھ فائن ٹیون کرنے سے واقف نہیں ہیں، تو بنیادی ٹیوٹوریل پر ایک نظر ڈالیں [یہاں](../training)!

</Tip>

اب آپ اپنے ماڈل کی تربیت شروع کرنے کے لیے تیار ہیں! Wav2Vec2 کو [`AutoModelForCTC`] کے ساتھ لوڈ کریں۔ `ctc_loss_reduction` پیرامیٹر کے ساتھ لاگو کرنے کے لیے کمی کی وضاحت کریں۔ ڈیفالٹ سمیشن کے بجائے اوسط استعمال کرنا اکثر بہتر ہوتا ہے:

```py
>>> from transformers import AutoModelForCTC, TrainingArguments, Trainer

>>> model = AutoModelForCTC.from_pretrained(
...     "facebook/wav2vec2-base",
...     ctc_loss_reduction="mean",
...     pad_token_id=processor.tokenizer.pad_token_id,
... )
```

اس وقت، صرف تین مراحل باقی ہیں:

1. [`TrainingArguments`] میں اپنے تربیتی ہائپر پیرامیٹر کی وضاحت کریں۔ صرف مطلوبہ پیرامیٹر `output_dir` ہے جو بتاتا ہے کہ آپ کے ماڈل کو کہاں محفوظ کرنا ہے۔ آپ اس ماڈل کو `push_to_hub=True` سیٹ کر کے حب کی طرف دھکیلیں گے (اپنا ماڈل اپ لوڈ کرنے کے لیے آپ کو Hugging Face میں سائن ان ہونا ہوگا)۔ ہر دور کے اختتام پر، [`Trainer`] WER کا جائزہ لے گا اور تربیت چیک پوائنٹ کو محفوظ کرے گا۔
2. ماڈل، ڈیٹاسیٹ، ٹوکنائزر، ڈیٹا کولیٹر، اور `compute_metrics` فنکشن کے ساتھ تربیتی دلائل [`Trainer`] پر منتقل کریں۔
3. [`~Trainer.train`] کو فائن ٹیون اپنے ماڈل پر کال کریں۔

```py
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_asr_mind_model",
...     per_device_train_batch_size=8,
...     gradient_accumulation_steps=2,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=2000,
...     gradient_checkpointing=True,
...     fp16=True,
...     train_sampling_strategy="group_by_length",
...     eval_strategy="steps",
...     per_device_eval_batch_size=8,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     load_best_model_at_end=True,
...     metric_for_best_model="wer",
...     greater_is_better=False,
...     push_to_hub=True,
...     report_to="trackio",
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=encoded_minds["train"],
...     eval_dataset=encoded_minds["test"],
...     processing_class=processor,
...     data_collator=data_collator,
...     compute_metrics=compute_metrics,
... )

>>> trainer.train()
```

تربیت مکمل ہونے کے بعد، اپنے ماڈل کو [`~transformers.Trainer.push_to_hub`] طریقہ کے ساتھ حب سے شیئر کریں تاکہ یہ ہر کسی کے لیے قابل رسائی ہو:

```py
>>> trainer.push_to_hub()
```

<Tip>

خودکار اسپیچ ریکگنیشن کے لیے فائن ٹیون a ماڈل کیسے بنایا جائے اس کی مزید گہرائی سے مثال کے لیے، اس بلاگ [post](https://huggingface.co/blog/fine-tune-wav2vec2-english) انگریزی ASR کے لیے اور یہ [post](https://huggingface.co/blog/fine-tune-xlsr-wav2vec2) کثیر لسانی ASR کے لیے دیکھیں۔

</Tip>

## انفرنس

بہت اچھا، اب جب کہ آپ نے ایک ماڈل ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں!

ایک آڈیو فائل لوڈ کریں جسے آپ انفرنس چلانا چاہتے ہیں۔ اگر آپ کو ضرورت ہو تو ماڈل کے نمونے لینے کی شرح سے ملنے کے لیے آڈیو فائل کے نمونے لینے کی شرح کو دوبارہ نمونہ بنانا یاد رکھیں!

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
>>> dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
>>> sampling_rate = dataset.features["audio"].sampling_rate
>>> audio_file = dataset[0]["audio"]["path"]
```

انفرنس کے لیے اپنے فائن ٹیونڈ ماڈل کو آزمانے کا آسان ترین طریقہ یہ ہے کہ اسے [`pipeline`] میں استعمال کریں۔ اپنے ماڈل کے ساتھ خودکار اسپیچ ریکگنیشن کے لیے ایک `pipeline` فوری بنائیں، اور اپنی آڈیو فائل اس میں بھیجیں:

```py
>>> from transformers import pipeline

>>> transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
>>> transcriber(audio_file)
{'text': 'I WOUD LIKE O SET UP JOINT ACOUNT WTH Y PARTNER'}
```

<Tip>

نقل مہذب ہے، لیکن یہ بہتر ہو سکتا ہے! اور بھی بہتر نتائج حاصل کرنے کے لیے مزید مثالوں پر اپنا ماڈل ٹھیک کرنے کی کوشش کریں!

</Tip>

اگر آپ چاہیں تو آپ `pipeline` کے نتائج کو دستی طور پر بھی نقل کر سکتے ہیں:

آڈیو فائل اور ٹرانسکرپشن کو پری پروسیس کرنے کے لیے ایک پروسیسر لوڈ کریں اور `input` کو PyTorch ٹینسر کے طور پر واپس کریں:

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
```

اپنے ان پٹس کو ماڈل پر منتقل کریں اور لاگٹس واپس کریں:

```py
>>> from transformers import AutoModelForCTC

>>> model = AutoModelForCTC.from_pretrained("stevhliu/my_awesome_asr_mind_model")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
```

سب سے زیادہ امکان کے ساتھ پیشین گوئی شدہ `input_ids` حاصل کریں، اور پیشین گوئی شدہ `input_ids` کو واپس متن میں ڈی کوڈ کرنے کے لیے پروسیسر کا استعمال کریں:

```py
>>> import torch

>>> predicted_ids = torch.argmax(logits, dim=-1)
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription
['I WOUL LIKE O SET UP JOINT ACOUNT WTH Y PARTNER']
```
