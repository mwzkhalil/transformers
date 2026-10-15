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

# متن سے تقریر

[[open-in-colab]]

ٹیکسٹ ٹو اسپیچ (TTS) متن سے قدرتی آواز والی اسپیچ بنانے کا کام ہے، جہاں اسپیچ کو متعدد زبانوں میں اور ایک سے زیادہ بولنے والوں کے لیے تیار کیا جاسکتا ہے۔ متعدد متن سے تقریرماڈلزفی الحال 🤗 میں دستیاب ہیں۔Transformers، جیسے [دیا](https://huggingface.co/docs/transformers/main/en/model_doc/dia), [CSM](https://huggingface.co/docs/transformers/main/en/model_doc/csm), [چھال](https://huggingface.co/docs/transformers/main/en/model_doc/bark), [MMS](https://huggingface.co/docs/transformers/main/en/model_doc/mms), [VITS](https://huggingface.co/docs/transformers/main/en/model_doc/vits)اور [SpeechT5](https://huggingface.co/docs/transformers/main/en/model_doc/speecht5).

آپ آسانی سے `"text-to-audio"` پائپ لائن (یا اس کا عرف - `"text-to-speech"`) استعمال کر کے آڈیو بنا سکتے ہیں۔ یہاں ایک مثال ہے کہ آپ `"text-to-speech"` پائپ لائن کو [CSM](https://huggingface.co/sesame/csm-1b) کے ساتھ کیسے استعمال کریں گے:

```python
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-audio", model="sesame/csm-1b")
>>> output = pipe("Hello from Sesame.")
```

یہاں ایک کوڈ کا ٹکڑا ہے جسے آپ نوٹ بک میں نتیجے میں آڈیو سننے کے لیے استعمال کر سکتے ہیں:

```python
>>> from IPython.display import Audio
>>> Audio(output["audio"], rate=output["sampling_rate"])
```

پہلے سے طے شدہ طور پر، CSM ایک بے ترتیب آواز استعمال کرتا ہے۔ آپ چیٹ ٹیمپلیٹ ڈکشنری کے حصے کے طور پر ایک حوالہ آڈیو فراہم کر کے صوتی کلوننگ کر سکتے ہیں:

```python
>>> import soundfile as sf
>>> import torch
>>> from datasets import Audio, load_dataset
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-audio", model="sesame/csm-1b")

>>> ds = load_dataset("hf-internal-testing/dailytalk-dummy", split="train")
>>> ds = ds.cast_column("audio", Audio(sampling_rate=24000))
>>> conversation = [
...     {
...         "role": "0",
...         "content": [
...             {"type": "text", "text": "What are you working on?"},
...             {"type": "audio", "path": ds[0]["audio"]["array"]},
...         ],
...     },
...     {"role": "0", "content": [{"type": "text", "text": "How much money can you spend?"}]},
... ]
>>> output = pipe(conversation)
```

کچھ ماڈلز، جیسے [Dia](https://huggingface.co/nari-labs/Dia-1.6B-0626)، غیر زبانی مواصلات جیسے ہنسنا، آہ بھرنا اور رونا، یا یہاں تک کہ موسیقی شامل کرنے کے لیے بھی مشروط کیا جا سکتا ہے۔ ذیل میں ایسی مثال ہے:

```python
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="nari-labs/Dia-1.6B-0626")
>>> text = "[S1] (clears throat) Hello! How are you? [S2] I'm good, thanks! How about you?"
>>> output = pipe(text)
```

نوٹ کریں کہ Dia انوکھی آوازوں کے درمیان بات چیت پیدا کرنے کے لیے اسپیکر کے ٹیگز جیسے [S1] اور [S2] کو بھی قبول کرتی ہے۔

مزید مثالوں کے لیے CSM اور دیگر پہلے سے تربیت یافتہ TTS ماڈلز کیا کر سکتے ہیں، ہمارے [آڈیو کورس](https://huggingface.co/learn/audio-course/chapter6/pre-trained_models) سے رجوع کریں۔

اگر آپ فائن ٹیون ایک TTS ماڈل تلاش کر رہے ہیں، تو صرف ٹیکسٹ ٹو اسپیچ ماڈلز فی الحال 🤗 Transformers میں دستیاب ہے [SpeechT5](https://huggingface.co/docs/transformers/main/en/model_doc/speecht5)، [FastSpeech2Conformer](https://huggingface.co/docs/transformers/main/en/model_doc/fastspeech2_conformer), [Dia](https://huggingface.co/docs/transformers/main/en/model_doc/dia) اور [CSM](https://huggingface.co/docs/transformers/main/en/model_doc/csm) اگرچہ مستقبل میں مزید شامل کیے جائیں گے۔ SpeechT5 پہلے سے تربیت یافتہ اسپیچ ٹو ٹیکسٹ اور ٹیکسٹ ٹو اسپیچ ڈیٹا کے امتزاج پر ہے، جس سے یہ متن اور تقریر دونوں کے ذریعے مشترکہ پوشیدہ نمائندگی کی ایک متحد جگہ سیکھ سکتا ہے۔ اس کا مطلب ہے کہ ایک ہی پہلے سے تربیت یافتہ ماڈل کو مختلف کاموں کے لیے ٹھیک بنایا جا سکتا ہے۔ مزید برآں، SpeechT5 ایکس ویکٹر سپیکر ایمبیڈنگ کے ذریعے متعدد اسپیکرز کو سپورٹ کرتا ہے۔

اس گائیڈ کا بقیہ حصہ یہ بتاتا ہے کہ کیسے:

1. فائن ٹیون [SpeechT5](https://huggingface.co/docs/transformers/main/en/model_doc/speecht5) جو اصل میں [VoxPopuli](https://huggingface.co/datasets/qmeeus/voxpopuli) ڈیٹاسیٹ کے ڈچ (`nl`) زبان کے ذیلی سیٹ پر انگریزی تقریر پر تربیت دی گئی تھی۔
2. انفرنس کے لیے اپنا بہتر ماڈل استعمال کریں دو طریقوں میں سے ایک میں: پائپ لائن کا استعمال کرتے ہوئے یا براہ راست۔

شروع کرنے سے پہلے، یقینی بنائیں کہ آپ کے پاس تمام ضروری لائبریریاں انسٹال ہیں:

```bash
pip install transformers datasets soundfile librosa torchcodec speechbrain accelerate
```

ماخذ سے 🤗Transformers انسٹال کریں کیونکہ ابھی تک تمام SpeechT5 خصوصیات کو سرکاری ریلیز میں ضم نہیں کیا گیا ہے:

```bash
pip install git+https://github.com/huggingface/transformers.git
```

<Tip>

اس گائیڈ پر عمل کرنے کے لیے آپ کو ایک GPU کی ضرورت ہوگی۔ اگر آپ ایک نوٹ بک میں کام کر رہے ہیں، تو یہ چیک کرنے کے لیے درج ذیل لائن چلائیں کہ آیا GPU دستیاب ہے:

```bash
!nvidia-smi
```

یا متبادل طور پر AMD GPUs کے لیے:

```bash
!rocm-smi
```

</Tip>

ہم آپ کی حوصلہ افزائی کرتے ہیں کہ آپ اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں تاکہ اپنا ماڈل اپ لوڈ کریں اور کمیونٹی کے ساتھ شیئر کریں۔ جب اشارہ کیا جائے، لاگ ان کرنے کے لیے اپنا ٹوکن درج کریں:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## ڈیٹاسیٹ لوڈ کریں۔

[VoxPopuli](https://huggingface.co/datasets/qmeeus/voxpopuli) ایک بڑے پیمانے پر کثیر لسانی اسپیچ کارپس ہے جس میں 2009-2020 یورپی پارلیمنٹ ایونٹ کی ریکارڈنگ سے حاصل کردہ ڈیٹا پر مشتمل ہے۔ اس میں 15 یورپی زبانوں کے لیے لیبل لگا ہوا آڈیو ٹرانسکرپشن ڈیٹا ہے۔ اس گائیڈ میں، ہم ڈچ زبان کا سب سیٹ استعمال کر رہے ہیں، بلا جھجھک کوئی دوسرا سب سیٹ منتخب کریں۔

نوٹ کریں کہ VoxPopuli یا کوئی دوسرا خودکار اسپیچ ریکگنیشن (ASR) ڈیٹاسیٹ TTS ماڈلز کی تربیت کے لیے موزوں ترین آپشن نہیں ہو سکتا۔ وہ خصوصیات جو اسے ASR کے لیے فائدہ مند بناتی ہیں، جیسے کہ ضرورت سے زیادہ پس منظر کا شور، TTS میں عام طور پر ناپسندیدہ ہوتا ہے۔ تاہم، اعلیٰ معیار، کثیر لسانی، اور کثیر اسپیکر والے TTS ڈیٹاسیٹس کو تلاش کرنا کافی مشکل ہو سکتا ہے۔

آئیے ڈیٹا لوڈ کرتے ہیں:

```py
>>> from datasets import load_dataset, Audio

>>> dataset = load_dataset("qmeeus/voxpopuli", "nl", split="train")
>>> len(dataset)
20968
```

20968 مثالیں فائن ٹیوننگ کے لیے کافی ہونی چاہئیں۔ SpeechT5 توقع کرتا ہے کہ آڈیو ڈیٹا کے نمونے لینے کی شرح 16 kHz ہوگی، لہذا یقینی بنائیں کہ ڈیٹاسیٹ میں موجود مثالیں اس ضرورت کو پورا کرتی ہیں:

```py
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
```

## ڈیٹا کو پہلے سے پروسیس کریں۔

آئیے مناسب پروسیسر کو استعمال کرنے اور لوڈ کرنے کے لیے ماڈل چیک پوائنٹ کی وضاحت کرتے ہوئے شروع کریں:

```py
>>> from transformers import SpeechT5Processor

>>> checkpoint = "microsoft/speecht5_tts"
>>> processor = SpeechT5Processor.from_pretrained(checkpoint)
```

### SpeechT5 ٹوکنائزیشن کے لیے ٹیکسٹ کلین اپ

ٹیکسٹ ڈیٹا کو صاف کرکے شروع کریں۔ متن پر کارروائی کرنے کے لیے آپ کو پروسیسر کے ٹوکنائزر حصے کی ضرورت ہوگی:

```py
>>> tokenizer = processor.tokenizer
```

ڈیٹا سیٹ کی مثالیں نقل کے ساتھ `text` خصوصیت پر مشتمل ہیں۔ نوٹ کریں کہ SpeechT5 ٹوکنائزر میں نمبرز کے لیے کوئی ٹوکن نہیں ہے، اس لیے متن میں کوئی بھی نمبر `<unk>` ٹوکنز میں تبدیل ہو جائے گا۔ اگر آپ کے ڈیٹاسیٹ میں نمبرز ہیں تو انہیں ان کی تحریری شکل میں معمول پر لانے پر غور کریں۔

چونکہ SpeechT5 کو انگریزی زبان میں تربیت دی گئی تھی، اس لیے یہ ڈچ ڈیٹاسیٹ میں بعض حروف کو نہیں پہچان سکتا۔ اگر اسی طرح چھوڑ دیا جائے تو یہ حروف `<unk>` ٹوکنز میں تبدیل ہو جائیں گے۔ تاہم، ڈچ میں، بعض حروف جیسے `à` کا استعمال حرفوں پر زور دینے کے لیے کیا جاتا ہے۔ متن کے معنی کو محفوظ رکھنے کے لیے، ہم اس حرف کو باقاعدہ `a` سے بدل سکتے ہیں۔

غیر تعاون یافتہ ٹوکنز کی شناخت کرنے کے لیے، `SpeechT5Tokenizer` کا استعمال کرتے ہوئے ڈیٹاسیٹ میں تمام منفرد حروف نکالیں جو کہ ٹوکن کے طور پر حروف کے ساتھ کام کرتے ہیں۔ ایسا کرنے کے لیے، `extract_all_chars` میپنگ فنکشن لکھیں جو تمام مثالوں سے نقل کو ایک سٹرنگ میں جوڑتا ہے اور اسے حروف کے سیٹ میں تبدیل کرتا ہے۔ `batched=True` اور `batch_size=-1` کو `dataset.map()` میں سیٹ کرنا یقینی بنائیں تاکہ نقشہ سازی کے فنکشن کے لیے تمام نقلیں ایک ساتھ دستیاب ہوں۔

```py
>>> def extract_all_chars(batch):
...     all_text = " ".join(batch["text"])
...     vocab = list(set(all_text))
...     return {"vocab": [vocab], "all_text": [all_text]}


>>> vocabs = dataset.map(
...     extract_all_chars,
...     batched=True,
...     batch_size=-1,
...     keep_in_memory=True,
...     remove_columns=dataset.column_names,
... )

>>> dataset_vocab = set(vocabs["vocab"][0])
>>> tokenizer_vocab = {k for k, _ in tokenizer.get_vocab().items()}
```

اب آپ کے پاس حروف کے دو سیٹ ہیں: ایک ڈیٹاسیٹ سے ذخیرہ الفاظ کے ساتھ اور دوسرا ٹوکنائزر سے ذخیرہ الفاظ کے ساتھ۔ ڈیٹا سیٹ میں کسی بھی غیر تعاون یافتہ حروف کی شناخت کرنے کے لیے، آپ ان دو سیٹوں کے درمیان فرق لے سکتے ہیں۔ نتیجے میں آنے والے سیٹ میں وہ حروف شامل ہوں گے جو ڈیٹاسیٹ میں ہیں لیکن ٹوکنائزر میں نہیں ہیں۔

```py
>>> dataset_vocab - tokenizer_vocab
{' ', 'à', 'ç', 'è', 'ë', 'í', 'ï', 'ö', 'ü'}
```

پچھلے مرحلے میں شناخت کیے گئے غیر تعاون یافتہ حروف کو ہینڈل کرنے کے لیے، ایک فنکشن کی وضاحت کریں جو ان حروف کو درست ٹوکنز میں نقش کرے۔ نوٹ کریں کہ خالی جگہیں پہلے ہی ٹوکنائزر میں `▁` سے بدل دی گئی ہیں اور انہیں الگ سے ہینڈل کرنے کی ضرورت نہیں ہے۔

```py
>>> replacements = [
...     ("à", "a"),
...     ("ç", "c"),
...     ("è", "e"),
...     ("ë", "e"),
...     ("í", "i"),
...     ("ï", "i"),
...     ("ö", "o"),
...     ("ü", "u"),
... ]


>>> def cleanup_text(inputs):
...     for src, dst in replacements:
...         inputs["text"] = inputs["text"].replace(src, dst)
...     return inputs


>>> dataset = dataset.map(cleanup_text)
```

اب جب کہ آپ نے متن میں خصوصی حروف کے ساتھ کام کر لیا ہے، اب وقت آ گیا ہے کہ آڈیو ڈیٹا پر توجہ مرکوز کی جائے۔

### مقررین

VoxPopuli ڈیٹاسیٹ میں متعدد اسپیکرز کی تقریر شامل ہے، لیکن ڈیٹاسیٹ میں کتنے اسپیکرز کی نمائندگی کی جاتی ہے؟ اس کا تعین کرنے کے لیے، ہم منفرد بولنے والوں کی تعداد اور مثالوں کی تعداد گن سکتے ہیں جن میں ہر اسپیکر ڈیٹاسیٹ میں تعاون کرتا ہے۔ ڈیٹاسیٹ میں کل 20,968 مثالوں کے ساتھ، یہ معلومات ہمیں ڈیٹا میں بولنے والوں اور مثالوں کی تقسیم کے بارے میں بہتر تفہیم فراہم کرے گی۔

```py
>>> from collections import defaultdict

>>> speaker_counts = defaultdict(int)

>>> for speaker_id in dataset["speaker_id"]:
...     speaker_counts[speaker_id] += 1
```

ہسٹوگرام بنا کر آپ اندازہ لگا سکتے ہیں کہ ہر اسپیکر کے لیے کتنا ڈیٹا ہے۔

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.hist(speaker_counts.values(), bins=20)
>>> plt.ylabel("Speakers")
>>> plt.xlabel("Examples")
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_speakers_histogram.png" alt="Speakers histogram"/>
</div>

ہسٹگرام سے پتہ چلتا ہے کہ ڈیٹاسیٹ میں تقریباً ایک تہائی بولنے والوں کی 100 سے کم مثالیں ہیں، جب کہ دس کے قریب بولنے والوں کی 500 سے زیادہ مثالیں ہیں۔ تربیت کی کارکردگی کو بہتر بنانے اور ڈیٹا سیٹ کو متوازن کرنے کے لیے، ہم 100 اور 400 کے درمیان مثالوں کے ساتھ ڈیٹا کو اسپیکر تک محدود کر سکتے ہیں۔

```py
>>> def select_speaker(speaker_id):
...     return 100 <= speaker_counts[speaker_id] <= 400


>>> dataset = dataset.filter(select_speaker, input_columns=["speaker_id"])
```

آئیے چیک کریں کہ کتنے بولنے والے باقی ہیں:

```py
>>> len(set(dataset["speaker_id"]))
42
```

آئیے دیکھتے ہیں کہ کتنی مثالیں باقی ہیں:

```py
>>> len(dataset)
9973
```

آپ کے پاس تقریباً 40 منفرد مقررین سے صرف 10,000 سے کم مثالیں رہ گئی ہیں، جو کافی ہونی چاہئیں۔

نوٹ کریں کہ چند مثالوں کے ساتھ کچھ بولنے والوں کی مثالیں لمبی ہونے کی صورت میں اصل میں زیادہ آڈیو دستیاب ہو سکتی ہے۔ تاہم، ہر اسپیکر کے لیے آڈیو کی کل مقدار کا تعین کرنے کے لیے پورے ڈیٹاسیٹ کے ذریعے اسکیننگ کی ضرورت ہوتی ہے، جو ایک وقت طلب عمل ہے جس میں ہر آڈیو فائل کو لوڈ کرنا اور ڈی کوڈ کرنا شامل ہے۔ اس طرح، ہم نے یہاں اس قدم کو چھوڑنے کا انتخاب کیا ہے۔

### اسپیکر ایمبیڈنگز

TTS ماڈل کو ایک سے زیادہ اسپیکر کے درمیان فرق کرنے کے لیے، آپ کو ہر مثال کے لیے ایک اسپیکر ایمبیڈنگ بنانے کی ضرورت ہوگی۔ اسپیکر ایمبیڈنگ ماڈل میں ایک اضافی ان پٹ ہے جو کسی مخصوص اسپیکر کی آواز کی خصوصیات کو پکڑتا ہے۔ یہ اسپیکر ایمبیڈنگز بنانے کے لیے، SpeechBrain سے پہلے سے تربیت یافتہ [spkrec-xvect-voxceleb](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) ماڈل استعمال کریں۔

ایک فنکشن `create_speaker_embedding()` بنائیں جو ایک ان پٹ آڈیو ویوفارم لیتا ہے اور ایک 512 عنصر ویکٹر کو آؤٹ پٹ کرتا ہے جس میں متعلقہ اسپیکر ایمبیڈنگ ہوتا ہے۔

```py
>>> import os
>>> import torch
>>> from speechbrain.inference.classifiers import EncoderClassifier
>>> from accelerate import Accelerator

>>> spk_model_name = "speechbrain/spkrec-xvect-voxceleb"
>>> device = Accelerator().device
>>> speaker_model = EncoderClassifier.from_hparams(
...     source=spk_model_name,
...     run_opts={"device": device},
...     savedir=os.path.join("/tmp", spk_model_name),
... )


>>> def create_speaker_embedding(waveform):
...     with torch.no_grad():
...         speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
...         speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
...         speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
...     return speaker_embeddings
```

یہ نوٹ کرنا ضروری ہے کہ `speechbrain/spkrec-xvect-voxceleb` ماڈل کو VoxCeleb ڈیٹا سیٹ سے انگریزی تقریر پر تربیت دی گئی تھی، جب کہ اس گائیڈ میں تربیت کی مثالیں ڈچ میں ہیں۔ اگرچہ ہمیں یقین ہے کہ یہ ماڈل ہمارے ڈچ ڈیٹاسیٹ کے لیے اب بھی معقول اسپیکر ایمبیڈنگس پیدا کرے گا، ہو سکتا ہے یہ مفروضہ تمام صورتوں میں درست نہ ہو۔

بہترین نتائج کے لیے، ہم تجویز کرتے ہیں کہ پہلے ٹارگٹ اسپیچ پر ایکس ویکٹر ماڈل تربیت دیں۔ یہ اس بات کو یقینی بنائے گا کہ ماڈل ڈچ زبان میں موجود منفرد آواز کی خصوصیات کو حاصل کرنے کے قابل ہے۔

### ڈیٹاسیٹ پر کارروائی ہو رہی ہے۔

آخر میں، آئیے ڈیٹا کو اس فارمیٹ میں پروسیس کریں جس کی ماڈل توقع ہے۔ ایک `prepare_dataset` فنکشن بنائیں جو ایک ہی مثال میں لے اور `SpeechT5Processor` آبجیکٹ کو ان پٹ ٹیکسٹ کو ٹوکنائز کرنے اور ٹارگٹ آڈیو کو لاگ میل سپیکٹروگرام میں لوڈ کرنے کے لیے استعمال کرے۔ اسے ایک اضافی ان پٹ کے طور پر اسپیکر ایمبیڈنگز کو بھی شامل کرنا چاہیے۔

```py
>>> def prepare_dataset(example):
...     audio = example["audio"]

...     example = processor(
...         text=example["text"],
...         audio_target=audio["array"],
...         sampling_rate=audio["sampling_rate"],
...         return_attention_mask=False,
...     )

...     # strip off the batch dimension
...     example["labels"] = example["labels"][0]

...     # use SpeechBrain to obtain x-vector
...     example["speaker_embeddings"] = create_speaker_embedding(audio["array"])

...     return example
```

ایک مثال کو دیکھ کر تصدیق کریں کہ پروسیسنگ درست ہے:

```py
>>> processed_example = prepare_dataset(dataset[0])
>>> list(processed_example.keys())
['input_ids', 'labels', 'stop_labels', 'speaker_embeddings']
```

اسپیکر ایمبیڈنگز 512 عنصر ویکٹر ہونی چاہئیں:

```py
>>> processed_example["speaker_embeddings"].shape
(512,)
```

لیبلز 80 میل ڈبوں کے ساتھ لاگ میل سپیکٹروگرام ہونے چاہئیں۔

```py
>>> import matplotlib.pyplot as plt

>>> plt.figure()
>>> plt.imshow(processed_example["labels"].T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_1.png" alt="Log-mel spectrogram with 80 mel bins"/>
</div>

ضمنی نوٹ: اگر آپ کو یہ اسپیکٹروگرام الجھا ہوا لگتا ہے، تو اس کی وجہ پلاٹ کے اوپری حصے میں کم تعدد اور اعلی تعدد رکھنے کے کنونشن سے آپ کی واقفیت ہوسکتی ہے۔ تاہم، جب matplotlib لائبریری کا استعمال کرتے ہوئے ایک تصویر کے طور پر سپیکٹروگرامس کی منصوبہ بندی کرتے ہیں، تو y-axis پلٹ جاتا ہے اور سپیکٹروگرام الٹا دکھائی دیتے ہیں۔

اب پروسیسنگ فنکشن کو پورے ڈیٹاسیٹ پر لگائیں۔ اس میں 5 سے 10 منٹ لگیں گے۔

```py
>>> dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names)
```

آپ کو ایک انتباہ نظر آئے گا جس میں کہا گیا ہے کہ ڈیٹاسیٹ میں کچھ مثالیں ماڈل زیادہ سے زیادہ ان پٹ لمبائی سے زیادہ لمبی ہیں (600 ٹوکنز)۔ ڈیٹا سیٹ سے ان مثالوں کو ہٹا دیں۔ یہاں ہم اور بھی آگے بڑھتے ہیں اور بڑے بیچ سائز کی اجازت دینے کے لیے ہم 200 سے زیادہ ٹوکن ہٹا دیتے ہیں۔

```py
>>> def is_not_too_long(input_ids):
...     input_length = len(input_ids)
...     return input_length < 200


>>> dataset = dataset.filter(is_not_too_long, input_columns=["input_ids"])
>>> len(dataset)
8259
```

اگلا، ایک بنیادی ٹرین/ٹیسٹ اسپلٹ بنائیں:

```py
>>> dataset = dataset.train_test_split(test_size=0.1)
```

### ڈیٹا کولیٹر

ایک بیچ میں متعدد مثالوں کو یکجا کرنے کے لیے، آپ کو حسب ضرورت ڈیٹا کولیٹر کی وضاحت کرنے کی ضرورت ہے۔ یہ کولیٹر پیڈنگ ٹوکنز کے ساتھ چھوٹے سلسلے کو پیڈ کرے گا، اس بات کو یقینی بناتا ہے کہ تمام مثالوں کی لمبائی ایک جیسی ہو۔ سپیکٹروگرام لیبلز کے لیے، پیڈڈ حصوں کو خصوصی قدر `-100` سے تبدیل کیا جاتا ہے۔ یہ خاص قدر ماڈل کو سپیکٹروگرام کے نقصان کا حساب لگاتے وقت سپیکٹروگرام کے اس حصے کو نظر انداز کرنے کی ہدایت کرتی ہے۔

```py
>>> from dataclasses import dataclass
>>> from typing import Any, Dict, List, Union


>>> @dataclass
... class TTSDataCollatorWithPadding:
...     processor: Any

...     def __call__(self, features: list[dict[str, Union[list[int], torch.Tensor]]]) -> dict[str, torch.Tensor]:
...         input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
...         label_features = [{"input_values": feature["labels"]} for feature in features]
...         speaker_features = [feature["speaker_embeddings"] for feature in features]

...         # collate the inputs and targets into a batch
...         batch = processor.pad(input_ids=input_ids, labels=label_features, return_tensors="pt")

...         # replace padding with -100 to ignore loss correctly
...         batch["labels"] = batch["labels"].masked_fill(batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100)

...         # not used during fine-tuning
...         del batch["decoder_attention_mask"]

...         # round down target lengths to multiple of reduction factor
...         if model.config.reduction_factor > 1:
...             target_lengths = torch.tensor([len(feature["input_values"]) for feature in label_features])
...             target_lengths = target_lengths.new(
...                 [length - length % model.config.reduction_factor for length in target_lengths]
...             )
...             max_length = max(target_lengths)
...             batch["labels"] = batch["labels"][:, :max_length]

...         # also add in the speaker embeddings
...         batch["speaker_embeddings"] = torch.tensor(speaker_features)

...         return batch
```

SpeechT5 میں، ماڈل کے ڈیکوڈر حصے میں ان پٹ کو ایک عنصر 2 سے کم کر دیا جاتا ہے۔ دوسرے الفاظ میں، یہ ہدف کی ترتیب سے ہر دوسرے ٹائم سٹیپ کو دور کر دیتا ہے۔ ڈیکوڈر پھر اس ترتیب کی پیشین گوئی کرتا ہے جو دوگنا لمبا ہوتا ہے۔ چونکہ اصل ہدف کی ترتیب کی لمبائی عجیب ہوسکتی ہے، اس لیے ڈیٹا کولیٹر اس بات کو یقینی بناتا ہے کہ بیچ کی زیادہ سے زیادہ لمبائی کو 2 کا ملٹیج نیچے کر دیا جائے۔

```py
>>> data_collator = TTSDataCollatorWithPadding(processor=processor)
```

## ماڈل کو تربیت دیں

پہلے سے تربیت یافتہ ماڈل کو اسی چیک پوائنٹ سے لوڈ کریں جیسا کہ آپ نے پروسیسر لوڈ کرنے کے لیے استعمال کیا تھا:

```py
>>> from transformers import SpeechT5ForTextToSpeech

>>> model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
```

`use_cache=True` آپشن گریڈینٹ چیک پوائنٹنگ کے ساتھ مطابقت نہیں رکھتا ہے۔ تربیت کے لیے اسے غیر فعال کریں۔

```py
>>> model.config.use_cache = False
```

تربیت کے دلائل کی وضاحت کریں۔ یہاں ہم تربیتی عمل کے دوران کسی بھی تشخیصی میٹرکس کا حساب نہیں لگا رہے ہیں۔ اس کے بجائے، ہم صرف نقصان کو دیکھیں گے:

```python
>>> from transformers import Seq2SeqTrainingArguments

>>> training_args = Seq2SeqTrainingArguments(
...     output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
...     per_device_train_batch_size=4,
...     gradient_accumulation_steps=8,
...     learning_rate=1e-5,
...     warmup_steps=500,
...     max_steps=4000,
...     gradient_checkpointing=True,
...     fp16=True,
...     eval_strategy="steps",
...     per_device_eval_batch_size=2,
...     save_steps=1000,
...     eval_steps=1000,
...     logging_steps=25,
...     report_to="trackio",
...     load_best_model_at_end=True,
...     greater_is_better=False,
...     label_names=["labels"],
...     push_to_hub=True,
... )
```

`Trainer` آبجیکٹ کو فوری بنائیں اور ماڈل، ڈیٹاسیٹ، اور ڈیٹا کولیٹر کو اس میں منتقل کریں۔

```py
>>> from transformers import Seq2SeqTrainer

>>> trainer = Seq2SeqTrainer(
...     args=training_args,
...     model=model,
...     train_dataset=dataset["train"],
...     eval_dataset=dataset["test"],
...     data_collator=data_collator,
...     processing_class=processor,
... )
```

اور اس کے ساتھ، آپ تربیت شروع کرنے کے لیے تیار ہیں! تربیت میں کئی گھنٹے لگیں گے۔ آپ کے GPU پر منحصر ہے، یہ ممکن ہے کہ جب آپ تربیت شروع کریں گے تو آپ کو CUDA "میموری سے باہر" کی غلطی کا سامنا کرنا پڑے گا۔ اس صورت میں، آپ 2 کے عوامل سے `per_device_train_batch_size` کو بتدریج کم کر سکتے ہیں اور تلافی کے لیے `gradient_accumulation_steps` کو 2x تک بڑھا سکتے ہیں۔

```py
>>> trainer.train()
```

پائپ لائن کے ساتھ اپنا چیک پوائنٹ استعمال کرنے کے لیے، پروسیسر کو چیک پوائنٹ کے ساتھ محفوظ کرنا یقینی بنائیں:

```py
>>> processor.save_pretrained("YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

فائنل ماڈل کو 🤗 حب کی طرف دھکیلیں:

```py
>>> trainer.push_to_hub()
```

## انفرنس

### انفرنس پائپ لائن کے ساتھ

بہت اچھا، اب جب کہ آپ نے ایک ماڈل ٹھیک کر لیا ہے، آپ اسے انفرنس کے لیے استعمال کر سکتے ہیں! سب سے پہلے، آئیے دیکھتے ہیں کہ آپ اسے متعلقہ پائپ لائن کے ساتھ کیسے استعمال کر سکتے ہیں۔ آئیے آپ کے چیک پوائنٹ کے ساتھ ایک `"text-to-speech"` پائپ لائن بنائیں:

```py
>>> from transformers import pipeline

>>> pipe = pipeline("text-to-speech", model="YOUR_ACCOUNT_NAME/speecht5_finetuned_voxpopuli_nl")
```

ڈچ میں متن کا ایک ٹکڑا منتخب کریں جسے آپ بیان کرنا چاہتے ہیں، جیسے:

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
```

پائپ لائن کے ساتھ SpeechT5 استعمال کرنے کے لیے، آپ کو اسپیکر ایمبیڈنگ کی ضرورت ہوگی۔ آئیے اسے ٹیسٹ ڈیٹاسیٹ میں ایک مثال سے حاصل کریں:

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

اب آپ ٹیکسٹ اور اسپیکر ایمبیڈنگز کو پائپ لائن میں منتقل کر سکتے ہیں، اور یہ باقی چیزوں کا خیال رکھے گا:

```py
>>> forward_params = {"speaker_embeddings": speaker_embeddings}
>>> output = pipe(text, forward_params=forward_params)
>>> output
{'audio': array([-6.82714235e-05, -4.26525949e-04,  1.06134125e-04, ...,
        -1.22392643e-03, -7.76011671e-04,  3.29112721e-04], dtype=float32),
 'sampling_rate': 16000}
```

اس کے بعد آپ نتیجہ سن سکتے ہیں:

```py
>>> from IPython.display import Audio
>>> Audio(output['audio'], rate=output['sampling_rate'])
```

### دستی طور پر انفرنس چلائیں۔

آپ پائپ لائن استعمال کیے بغیر انفرنس نتائج حاصل کر سکتے ہیں، تاہم، مزید اقدامات کی ضرورت ہوگی۔

🤗 حب سے ماڈل لوڈ کریں:

```py
>>> model = SpeechT5ForTextToSpeech.from_pretrained("YOUR_ACCOUNT/speecht5_finetuned_voxpopuli_nl")
```

ٹیسٹ ڈیٹاسیٹ سے ایک مثال چنیں اور اسپیکر ایمبیڈنگ حاصل کریں۔

```py
>>> example = dataset["test"][304]
>>> speaker_embeddings = torch.tensor(example["speaker_embeddings"]).unsqueeze(0)
```

ان پٹ ٹیکسٹ کی وضاحت کریں اور اسے ٹوکنائز کریں۔

```py
>>> text = "hallo allemaal, ik praat nederlands. groetjes aan iedereen!"
>>> inputs = processor(text=text, return_tensors="pt")
```

اپنے ماڈل کے ساتھ ایک سپیکٹروگرام بنائیں:

```py
>>> spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
```

اگر آپ چاہیں تو سپیکٹروگرام کا تصور کریں:

```py
>>> plt.figure()
>>> plt.imshow(spectrogram.T)
>>> plt.show()
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/tts_logmelspectrogram_2.png" alt="Generated log-mel spectrogram"/>
</div>

آخر میں، سپیکٹروگرام کو آواز میں تبدیل کرنے کے لیے ووکوڈر کا استعمال کریں۔

```py
>>> with torch.no_grad():
...     speech = vocoder(spectrogram)

>>> from IPython.display import Audio

>>> Audio(speech.numpy(), rate=16000)
```

ہمارے تجربے میں، اس ماڈل سے تسلی بخش نتائج حاصل کرنا مشکل ہو سکتا ہے۔ اسپیکر ایمبیڈنگ کا معیار ایک اہم عنصر معلوم ہوتا ہے۔ چونکہ SpeechT5 انگریزی x-vectors کے ساتھ پہلے سے تربیت یافتہ تھا، اس لیے انگریزی اسپیکر ایمبیڈنگز کا استعمال کرتے وقت یہ بہترین کارکردگی کا مظاہرہ کرتا ہے۔ اگر ترکیب شدہ تقریر خراب لگتی ہے تو، مختلف اسپیکر ایمبیڈنگ استعمال کرنے کی کوشش کریں۔

تربیت کا دورانیہ بڑھانے سے نتائج کے معیار میں بھی اضافہ ہونے کا امکان ہے۔ اس کے باوجود، تقریر واضح طور پر انگریزی کی بجائے ڈچ ہے، اور یہ اسپیکر کی آواز کی خصوصیات کو حاصل کرتی ہے (مثال میں اصل آڈیو سے موازنہ کریں)۔ ایک اور چیز جس کے ساتھ تجربہ کرنا ہے وہ ہے ماڈل کی ترتیب۔ مثال کے طور پر، یہ دیکھنے کے لیے `config.reduction_factor = 1` استعمال کرنے کی کوشش کریں کہ آیا اس سے نتائج بہتر ہوتے ہیں۔

آخر میں، اخلاقی تحفظات پر غور کرنا ضروری ہے۔ اگرچہ TTS ٹکنالوجی میں متعدد مفید ایپلی کیشنز ہیں، لیکن اس کا استعمال بدنیتی پر مبنی مقاصد کے لیے بھی کیا جا سکتا ہے، جیسے کہ کسی کے علم یا رضامندی کے بغیر اس کی آواز کی نقالی کرنا۔ براہ کرم TTS کا استعمال عقلمندی اور ذمہ داری سے کریں۔
