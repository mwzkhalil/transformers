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

# فوری آغاز

[[open-in-colab]]

Transformers کو تیز اور آسان استعمال کے لیے بنایا گیا ہے تاکہ ہر شخص ٹرانسفارمر ماڈلز کے ساتھ سیکھنا یا کچھ بنانا شروع کر سکے۔

صارف کے سامنے ماڈل بنانے کے لیے صرف تین بنیادی کلاسیں اور انفرنس یا تربیت کے لیے دو APIs ہیں۔ اس مختصر رہنمائی میں Transformers کی اہم خصوصیات متعارف کرائی گئی ہیں اور بتایا گیا ہے کہ آپ کیسے:

- پہلے سے تربیت یافتہ ماڈل لوڈ کریں
- [`Pipeline`] سے انفرنس چلائیں
- [`Trainer`] سے ماڈل کی فائن ٹیوننگ کریں

## ابتدائی ترتیب

شروع میں ہم Hugging Face پر [اکاؤنٹ](https://hf.co/join) بنانے کی تجویز دیتے ہیں۔ اکاؤنٹ کے ذریعے آپ [Hub](https://hf.co/docs/hub/index) پر ورژن کنٹرول والے ماڈلز، ڈیٹاسیٹس اور [Spaces](https://hf.co/spaces) محفوظ اور استعمال کر سکتے ہیں۔ Hub دریافت اور تعمیر کا اشتراکی پلیٹ فارم ہے۔

[User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) بنائیں اور اپنے اکاؤنٹ میں لاگ اِن کریں۔

<hfoptions id="authenticate">
<hfoption id="notebook">

جب لاگ اِن کرنے کا کہا جائے تو اپنا User Access Token [`~huggingface_hub.notebook_login`] میں درج کریں۔

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
<hfoption id="CLI">

یقینی بنائیں کہ [huggingface_hub[cli]](https://huggingface.co/docs/huggingface_hub/guides/cli#getting-started) پیکیج نصب ہے، پھر درج ذیل کمانڈ چلائیں۔ لاگ اِن کا اشارہ آنے پر اپنا User Access Token درج کریں۔

```bash
hf auth login
```

</hfoption>
</hfoptions>

PyTorch نصب کریں۔

```bash
# remove ! if installing from the CLI
!pip install torch
```

پھر Transformers کا تازہ ترین ورژن اور Hugging Face کے ماحولیاتی نظام کی کچھ اضافی لائبریریاں نصب کریں، جو ڈیٹاسیٹس اور وژن ماڈلز تک رسائی، تربیت کا جائزہ لینے اور بڑے ماڈلز کی تربیت کو بہتر بنانے کے کام آتی ہیں۔

```bash
# remove ! if installing from the CLI
!pip install -U transformers datasets evaluate accelerate timm
```

## ایجنٹ کی مہارتیں

[Hugging Face Skills](https://github.com/huggingface/skills) آپ کے coding agent کو Transformers کے ساتھ کام کرنا سکھاتی ہیں۔ تربیت کا اسکرپٹ خود لکھنے کے بجائے agent سے ماڈل کی فائن ٹیوننگ کروائیں اور skill کو اس کی رہنمائی کرنے دیں۔

<hfoptions id="install">
<hfoption id="Claude Code">

```sh
/plugin marketplace add huggingface/skills
/plugin install hf-cli@huggingface/skills
```

</hfoption>
<hfoption id="Codex">

```sh
codex plugin marketplace add huggingface/skills
```

پھر Codex میں `/plugins` چلائیں اور `huggingface/skills` ریپوزٹری سے ایک skill نصب کریں۔

</hfoption>
</hfoptions>

Cursor اور Gemini جیسے مزید پلیٹ فارمز کے لیے [Installation](https://github.com/huggingface/skills#installation) رہنمائی دیکھیں۔

skill نصب ہونے کے بعد اسے اپنے agent کے لیے ہدایات میں شامل کریں۔

```md
Use the HF Trainer skill to fine-tune RT-DETRv2 on [Lekim89/sportsmot](https://huggingface.co/datasets/Lekim89/sportsmot) for basketball player tracking.
```

## پہلے سے تربیت یافتہ ماڈلز

ہر پہلے سے تربیت یافتہ ماڈل تین بنیادی کلاسوں سے وراثت حاصل کرتا ہے۔

| **کلاس** | **وضاحت** |
|---|---|
| [`PreTrainedConfig`] | ماڈل کی خصوصیات، مثلاً attention heads کی تعداد یا vocabulary کا حجم، متعین کرنے والی فائل۔ |
| [`PreTrainedModel`] | configuration فائل میں درج خصوصیات سے متعین ہونے والا ماڈل (یا آرکیٹیکچر)۔ پہلے سے تربیت یافتہ ماڈل صرف خام hidden states واپس کرتا ہے۔ کسی مخصوص کام کے لیے مناسب model head استعمال کریں تاکہ ان سے بامعنی نتیجہ حاصل ہو (مثلاً [`LlamaModel`] کے مقابلے میں [`LlamaForCausalLM`])۔ |
| Preprocessor | خام input (متن، تصاویر، آڈیو یا ملٹی موڈل مواد) کو ماڈل کے لیے عددی input میں بدلنے والی کلاس۔ مثلاً [`PreTrainedTokenizer`] متن کو tensors میں اور [`ImageProcessingMixin`] پکسلز کو tensors میں بدلتا ہے۔ |

ماڈلز اور preprocessors لوڈ کرنے کے لیے ہم [AutoClass](https://huggingface.co/docs/transformers/main/en/model_doc/auto) API کی تجویز دیتے ہیں۔ یہ پہلے سے تربیت یافتہ weights اور configuration فائل کے نام یا path سے ہر کام کے لیے مناسب آرکیٹیکچر اور مشین لرننگ فریم ورک خود معلوم کر لیتی ہے۔

Hub سے weights اور configuration فائل ماڈل اور preprocessor کلاس میں لوڈ کرنے کے لیے [`~PreTrainedModel.from_pretrained`] استعمال کریں۔

ماڈل لوڈ کرتے وقت، مناسب لوڈنگ کے لیے درج ذیل parameters مقرر کریں۔

- `device_map="auto"` ماڈل کے weights پہلے آپ کی تیز ترین ڈیوائس پر خودکار طور پر رکھتا ہے۔
- `dtype="auto"` ماڈل کے weights کو براہِ راست اسی data type میں بناتا ہے جس میں وہ محفوظ ہیں۔ اس سے weights کو دو بار لوڈ کرنے سے بچا جا سکتا ہے (PyTorch طے شدہ طور پر انہیں `torch.float32` میں لوڈ کرتا ہے)۔

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
```

ٹوکنائزر (tokenizer) سے متن کو ٹوکنز میں بدلیں اور PyTorch tensors حاصل کریں۔ انفرنس سے پہلے tokenized inputs کو ماڈل ہی کی ڈیوائس پر منتقل کریں تاکہ ماڈل انہیں استعمال کر سکے۔

```py
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="pt").to(model.device)
```

اب ماڈل انفرنس یا تربیت کے لیے تیار ہے۔

انفرنس کے لیے tokenized inputs کو [`~GenerationMixin.generate`] دیں تاکہ متن تیار ہو۔ پھر [`~PreTrainedTokenizerBase.batch_decode`] سے token IDs کو دوبارہ متن میں بدلیں۔

```py
generated_ids = model.generate(**model_inputs, max_length=30)
tokenizer.batch_decode(generated_ids)[0]
'<s> The secret to baking a good cake is 100% in the preparation. There are so many recipes out there,'
```

> [!TIP]
> ماڈل کی فائن ٹیوننگ سیکھنے کے لیے براہِ راست [Trainer](#trainer) حصے پر جائیں۔

## Pipeline

[`Pipeline`] کلاس پہلے سے تربیت یافتہ ماڈل سے انفرنس چلانے کا سب سے آسان طریقہ ہے۔ یہ متن بنانے، تصویر کو حصوں میں تقسیم کرنے، خودکار گفتار کی شناخت، دستاویز سے سوالوں کے جواب دینے اور کئی دوسرے کاموں کی معاونت کرتی ہے۔

> [!TIP]
> دستیاب کاموں کی مکمل فہرست کے لیے [Pipeline](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines) API حوالہ دیکھیں۔

ایک [`Pipeline`] آبجیکٹ بنائیں اور کام منتخب کریں۔ طے شدہ طور پر [`Pipeline`] اس کام کے لیے ایک طے شدہ پہلے سے تربیت یافتہ ماڈل ڈاؤن لوڈ کرکے cache میں رکھتی ہے۔ کوئی خاص ماڈل منتخب کرنے کے لیے اس کا نام `model` parameter میں دیں۔

<hfoptions id="pipeline-tasks">
<hfoption id="text generation">

انفرنس کے لیے دستیاب accelerator خودکار طور پر معلوم کرنے کے لیے [`Accelerator`] استعمال کریں۔

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device=device)
```

مزید متن بنانے کے لیے [`Pipeline`] کو کچھ ابتدائی متن دیں۔

```py
pipe("The secret to baking a good cake is ", max_length=50)
[{'generated_text': 'The secret to baking a good cake is 100% in the batter. The secret to a great cake is the icing.\nThis is why we’ve created the best buttercream frosting reci'}]
```

</hfoption>
<hfoption id="image segmentation">

انفرنس کے لیے دستیاب accelerator خودکار طور پر معلوم کرنے کے لیے [`Accelerator`] استعمال کریں۔

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic", device=device)
```

[`Pipeline`] کو تصویر دیں، جو URL یا تصویر کا مقامی path ہو سکتی ہے۔

<div class="flex justify-center">
   <img src="https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png"/>
</div>

```py
segments = pipeline("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
segments[0]["label"]
'bird'
segments[1]["label"]
'bird'
```

</hfoption>
<hfoption id="automatic speech recognition">

انفرنس کے لیے دستیاب accelerator خودکار طور پر معلوم کرنے کے لیے [`Accelerator`] استعمال کریں۔

```py
from transformers import pipeline
from accelerate import Accelerator

device = Accelerator().device

pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device=device)
```

[`Pipeline`] کو آڈیو فائل دیں۔

```py
pipeline("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac")
{'text': ' He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered flour-fatten sauce.'}
```

</hfoption>
</hfoptions>

## Trainer

[`Trainer`]، PyTorch ماڈلز کی تربیت اور جانچ کے لیے مکمل loop ہے۔ یہ تربیتی loop خود لکھنے کے کئی بار بار آنے والے مراحل سنبھالتا ہے، تاکہ آپ جلد تربیت شروع کرکے تربیت کے ڈیزائن پر توجہ دے سکیں۔ dataset سے batches بنانے کے لیے آپ کو صرف ایک ماڈل، dataset، preprocessor اور data collator درکار ہیں۔

تربیت کا عمل اپنی ضرورت کے مطابق بنانے کے لیے [`TrainingArguments`] کلاس استعمال کریں۔ اس میں تربیت، جانچ اور دوسرے کاموں کے لیے کئی اختیارات ہیں۔ اپنی ضروریات کے مطابق batch size، learning rate، mixed precision، torch.compile اور دوسری خصوصیات آزمائیں۔ ابتدائی baseline جلد حاصل کرنے کے لیے تربیت کے طے شدہ parameters بھی استعمال کیے جا سکتے ہیں۔

تربیت کے لیے ماڈل، ٹوکنائزر اور dataset لوڈ کریں۔

```py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
```

متن کو tokenize کرکے PyTorch tensors میں بدلنے کے لیے function بنائیں۔ پھر [`~datasets.Dataset.map`] طریقے سے اسے پورے dataset پر لاگو کریں۔

```py
def tokenize_dataset(dataset):
    return tokenizer(dataset["text"])
dataset = dataset.map(tokenize_dataset, batched=True)
```

data کے batches بنانے کے لیے data collator لوڈ کریں اور اسے ٹوکنائزر دیں۔

```py
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

اس کے بعد تربیت کی خصوصیات اور hyperparameters کے ساتھ [`TrainingArguments`] مقرر کریں۔

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="distilbert-rotten-tomatoes",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=True,
)
```

آخر میں یہ تمام اجزا [`Trainer`] کو دیں اور تربیت شروع کرنے کے لیے [`~Trainer.train`] چلائیں۔

```py
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()
```

اپنا ماڈل اور ٹوکنائزر Hub پر شیئر کرنے کے لیے [`~Trainer.push_to_hub`] استعمال کریں۔

```py
trainer.push_to_hub()
```

مبارک ہو، آپ نے ابھی Transformers کے ساتھ اپنا پہلا ماڈل تربیت دیا ہے!

## اگلے اقدامات

اب جب کہ آپ Transformers اور اس کی سہولتوں کو بہتر سمجھتے ہیں، اپنی دلچسپی کے موضوعات سیکھتے اور آزماتے رہیں۔

- **بنیادی کلاسیں**: configuration، model اور processor کلاسوں کے بارے میں مزید جانیں۔ اس سے آپ کو ماڈلز بنانے اور حسبِ ضرورت بدلنے، مختلف inputs (آڈیو، تصاویر، ملٹی موڈل مواد) کو preprocess کرنے اور ماڈل شیئر کرنے کا طریقہ سمجھنے میں مدد ملے گی۔
- **انفرنس**: [`Pipeline`]، انفرنس، LLMs کے ساتھ گفتگو، agents، اور اپنے مشین لرننگ فریم ورک اور hardware کے ذریعے انفرنس بہتر بنانے کے بارے میں مزید جانیں۔
- **تربیت**: [`Trainer`]، تقسیم شدہ تربیت اور مخصوص hardware پر تربیت بہتر بنانے کا مزید مطالعہ کریں۔
- **Quantization**: weights کو کم bits میں ظاہر کرکے memory اور storage کی ضروریات گھٹائیں اور انفرنس تیز کریں۔
- **وسائل**: کسی مخصوص کام کے لیے ماڈل کی تربیت اور انفرنس کے شروع سے آخر تک کے طریقے ڈھونڈ رہے ہیں؟ task recipes دیکھیں!

بنیادی معلومات کے لیے [Transformers کا تعارف](./index) یا [تنصیب](./installation) دوبارہ دیکھیں۔
