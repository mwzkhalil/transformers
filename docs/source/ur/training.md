<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# فائن ٹیوننگ

فائن ٹیوننگ ایک بڑے پہلے سے تربیت یافتہ ماڈل کو کسی کام یا ڈومین کے لیے مخصوص چھوٹے ڈیٹا سیٹ پر تربیت دینا جاری رکھتا ہے۔ مثال کے طور پر، کوڈنگ کی مثالوں کے ڈیٹاسیٹ پر فائن ٹیوننگ ماڈل کو کوڈنگ میں بہتر ہونے میں مدد کرتا ہے۔ فائن ٹیوننگ پہلے سے تربیت کرنے کے مترادف ہے سوائے اس کے کہ آپ بے ترتیب وزن کے ساتھ شروع نہ کریں۔ اس کے لیے بہت کم حساب کتاب، ڈیٹا اور وقت درکار ہوتا ہے۔

ذیل میں ٹیوٹوریل فائن ٹیوننگ ایک بڑی زبان ماڈل کے ساتھ [`Trainer`] کے ذریعے چلتا ہے۔

اپنے صارف کے ٹوکن کے ساتھ اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں تاکہ آپ اپنے فائن ٹیونڈ ماڈل کو حب تک لے جائیں۔

```py
from huggingface_hub import login

login()
```

## ٹوکنائزیشن

ایک ڈیٹاسیٹ لوڈ کریں اور [tokenize](./fast_tokenizers) ٹیکسٹ کالم ماڈل ٹرینوں پر (`horoscope` نیچے ڈیٹا سیٹ میں)

<iframe
  src="https://huggingface.co/datasets/karthiksagarn/astro_horoscope/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

ٹوکنائزر ماڈل ان پٹ، `input_ids` اور `attention_mask` تخلیق کرتا ہے۔ ماڈل کا فارورڈ طریقہ صرف `input_ids` اور `attention_mask` کو قبول کرتا ہے، لہذا ٹوکنائزیشن کے بعد `horoscope` جیسے کالم چھوڑنے کے لیے `remove_columns` سیٹ کریں۔

- لمبے سلسلے کو ایک مخصوص زیادہ سے زیادہ لمبائی تک تراشنے کے لیے `truncation=True` اور `max_length` سیٹ کریں۔
- ماڈل کا اندازہ کرنے کے لیے ٹیسٹ اسپلٹ بنانے کے لیے [`~datasets.train_test_split`] طریقہ استعمال کریں۔

```py
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
dataset = dataset.train_test_split(test_size=0.1)
```

ڈیٹا کولیٹر ماڈل کو پروسیس کرنے کے لیے ڈیٹا سیٹ کے نمونوں کو بیچوں میں جمع کرتا ہے۔ [`DataCollatorForLanguageModeling`] *متحرک طور پر* ڈیٹاسیٹ میں ہر ترتیب کو ایک ہی لمبائی میں پیڈ کرنے کے بجائے ہر بیچ کو اس بیچ میں سب سے طویل ترتیب میں پیڈ کرتا ہے۔ یہ غیر ضروری پیڈنگ ٹوکن کمپیوٹنگ سے گریز کرکے کمپیوٹ اور میموری کو بچاتا ہے۔

- تصادفی طور پر ٹوکن ماسک کرنے سے بچنے کے لیے `mlm=False` سیٹ کریں۔

```py
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

## ماڈل لوڈ ہو رہا ہے

پہلے سے تربیت یافتہ چیک پوائنٹ سے فائن ٹیون لوڈ کریں (لوڈنگ ماڈلز کے بارے میں مزید تفصیلات کے لیے [لوڈنگ ماڈلز](https://huggingface.co/docs/transformers/main/en/models) گائیڈ دیکھیں)۔

- ان کی محفوظ کردہ قسم میں وزن لوڈ کرنے کے لیے `dtype="auto"` سیٹ کریں۔ اس کے بغیر، PyTorch وزن کو `torch.float32` میں لوڈ کرتا ہے، جو میموری کے استعمال کو دوگنا کر دیتا ہے اگر وزن اصل میں `torch.bfloat16` ہو۔

```py
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")
```

## تربیت کی ترتیب

[`TrainingArguments`] ٹریننگ رن کو حسب ضرورت بنانے کے لیے تمام اختیارات فراہم کرتا ہے۔ یہاں صرف سب سے عام دلائل کا احاطہ کیا گیا ہے۔ باقی ہر چیز میں معقول ڈیفالٹس ہیں یا صرف تقسیم شدہ تربیت جیسے مخصوص منظرناموں سے متعلق ہیں۔ دلائل کی مکمل فہرست کے لیے [`TrainingArguments`] API دستاویزات دیکھیں۔

<hfoptions id="training-args">
<hfoption id="training duration">

- `num_train_epochs` اور `per_device_train_batch_size` کنٹرول ٹریننگ کا دورانیہ اور بیچ سائز۔ `learning_rate` آپٹیمائزر کے لیے ابتدائی سیکھنے کی شرح متعین کرتا ہے۔

</hfoption>
<hfoption id="training optimizations">

- اگر آپ کا ہارڈویئر اس کو سپورٹ کرتا ہے تو تیز مخلوط صحت سے متعلق تربیت کے لیے `bf16=True` سیٹ کریں (Ampere+ GPUs)۔ بصورت دیگر، پرانے ہارڈ ویئر پر واپس `fp16=True` پر جائیں۔
- `gradient_accumulation_steps` وزن کو اپ ڈیٹ کرنے سے پہلے ایک سے زیادہ فارورڈ پاسز پر گریڈینٹ جمع کر کے ایک بڑے موثر بیچ سائز کی نقل کرتا ہے۔
- `gradient_checkpointing` ٹریڈز بیک ورڈ پاس کے دوران انٹرمیڈیٹ ایکٹیویشنز کو ذخیرہ کرنے کے بجائے دوبارہ کمپیوٹنگ کرکے میموری کے لیے کمپیوٹ کرتے ہیں۔ میموری اور رفتار کے درمیان سمجھوتہ کے لیے، دیکھیں [جزوی چیک پوائنٹنگ](https://huggingface.co/docs/transformers/main/en/grad_checkpointing#partial-checkpointing)۔
- جب `gradient_checkpointing=True`، `gradient_checkpointing_kwargs={"offload": True}` سیٹ کریں تاکہ محفوظ شدہ ایکٹیویشنز کو پن کی گئی ہوسٹ میموری میں رکھیں اور طویل ترتیب کے لیے GPU میموری کے استعمال کو کم کریں۔ ٹریڈ آف کے لیے [محفوظ کردہ ایکٹیویشنز کو آف لوڈ کرنا](https://huggingface.co/docs/transformers/main/en/grad_checkpointing#offloading-the-saved-activations) دیکھیں۔
- `train_sampling_strategy="group_by_length"` کو ایک جیسی لمبائی والی مثالوں کے بیچ پر سیٹ کریں اور پیڈنگ کم کریں۔ پروسیسر پر مبنی ملٹی موڈل ڈیٹاسیٹس اور پہلے سے گنتی کی لمبائی کے لیے [گروپ نمونے بلحاظ لمبائی](https://huggingface.co/docs/transformers/main/en/trainer_recipes#group-samples-by-length) دیکھیں۔

</hfoption>
<hfoption id="evaluation and checkpointing">

- `eval_strategy` اور `save_strategy` اس بات کا تعین کرتے ہیں کہ تربیت کے دوران ماڈل کا اندازہ کب کرنا ہے اور چیک پوائنٹ کو کب بچانا ہے۔
- ٹریننگ ختم ہونے پر `load_best_model_at_end` بہترین چیک پوائنٹ لوڈ کرتا ہے۔ اسے سیٹ کرنے کے لیے `eval_strategy` کی ضرورت ہے۔

</hfoption>
<hfoption id="logging">

- `logging_steps` کنٹرول کرتا ہے کہ تربیت کے دوران نقصان کو کتنی بار اپ ڈیٹ کرنا ہے اور واپس کرنا ہے۔

</hfoption>
</hfoptions>

```py
training_args = TrainingArguments(
    output_dir="qwen3-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

## تربیت

تمام ضروری اجزاء کے ساتھ ایک [`Trainer`] مثال بنائیں، پھر شروع کرنے کے لیے [`~Trainer.train`] کو کال کریں۔

```py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
trainer.push_to_hub()
```

[`~Trainer.push_to_hub`] فائن ٹیونڈ وزن، جنریشن کنفیگریشن، ٹوکنائزر، اور ماڈل کنفیگریشن کو حب پر اپ لوڈ کرتا ہے۔

## اگلے اقدامات

- ٹرینر کی عام خصوصیات کی کم سے کم کام کرنے والی مثالوں جیسے حسب ضرورت نقصان کے فنکشنز، یادداشت کی موثر تشخیص، چیک پوائنٹنگ اور مزید کے لیے [ٹرینر فیچرز](https://huggingface.co/docs/transformers/main/en/trainer_recipes) گائیڈ پڑھیں۔
- نئی اور حسب ضرورت فنکشنلٹیز کو سپورٹ کرنے کے لیے [`Trainer`] طریقوں کو سب کلاس کرنے کا طریقہ سیکھنے کے لیے [سب کلاسنگ ٹرینر میتھڈز](https://huggingface.co/docs/transformers/main/en/trainer_customize) گائیڈ پڑھیں۔
- لاگنگ، جلدی رکنے، اور دیگر حسب ضرورت رویے کے لیے تربیتی ایونٹس میں شامل ہونے کا طریقہ سیکھنے کے لیے [Callbacks](https://huggingface.co/docs/transformers/main/en/trainer_callbacks) گائیڈ پڑھیں۔
- نمونوں کو بیچوں میں جمع کرنے کے طریقے کو کس طرح اپنی مرضی کے مطابق بنانا سیکھنے کے لیے [ڈیٹا کولیٹرز](https://huggingface.co/docs/transformers/main/en/data_collators) گائیڈ پڑھیں۔
- کرائے کے GPU پر [Hugging Face جابز](https://huggingface.co/docs/hub/jobs-training) کے ساتھ ٹریننگ اسکرپٹ چلائیں۔
- مختلف متن، آڈیو، وژن، اور ملٹی موڈل کاموں پر اضافی تربیتی مثالوں کے لیے [notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) یا **وسائل > Task Recipes** سیکشن کو براؤز کریں۔
