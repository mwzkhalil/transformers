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

# پیرامیٹر مؤثر فائن ٹیوننگ

[پیرامیٹر موثرفائن ٹیوننگ(PEFT)](https://huggingface.co/docs/peft/index)صرف طریقےفائن ٹیوناضافی کی ایک چھوٹی سی تعدادماڈلپیرامیٹرز (اڈیپٹر) a کے اوپرپہلے سے تربیت یافتہ ماڈل. چونکہ صرف اڈاپٹر کے پیرامیٹرز کو اپ ڈیٹ کیا جاتا ہے، اس لیے آپٹیمائزر بہت کم گریڈینٹ اور اسٹیٹس کو ٹریک کرتا ہے، جس سے میموری کے استعمال کو نمایاں طور پر کم کیا جاتا ہے۔ اڈاپٹر ہلکے ہوتے ہیں، جو انہیں شیئر کرنے، اسٹور کرنے اور لوڈ کرنے میں آسان بناتے ہیں۔

Transformers براہ راست PEFT لائبریری کے ساتھ [`~integrations.PeftAdapterMixin`] کے ذریعے ضم ہوتا ہے، تمام [`PreTrainedModel`] کلاسوں میں شامل ہوتا ہے۔ آپ اپنے ماڈل کو علیحدہ [`~peft.PeftModel`] میں لپیٹے بغیر اڈاپٹر لوڈ، شامل، ٹرین، سوئچ، اور ڈیلیٹ کر سکتے ہیں۔ تمام غیر فوری سیکھنے والے PEFT طریقے معاون ہیں (LoRA, IA3, AdaLoRA)۔ پرامپٹ ٹیوننگ اور پریفکس ٹیوننگ جیسے پرامپٹ پر مبنی طریقوں کے لیے [PEFT لائبریری](https://huggingface.co/docs/peft/index) براہ راست استعمال کرنے کی ضرورت ہوتی ہے۔

شروع کرنے کے لیے PEFT انسٹال کریں۔ انضمام کے لیے `peft >= 0.20.0` درکار ہے۔

```shell
pip install -U peft
```

## ایک اڈاپٹر شامل کریں۔

ایک PEFT کنفیگریشن بنائیں، جیسے [`~peft.LoraConfig`] مثال کے طور پر، اور اسے ماڈل کے ساتھ [`~integrations.PeftAdapterMixin.add_adapter`] کے ساتھ منسلک کریں۔

```py
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model.add_adapter(lora_config, adapter_name="my_adapter")
```

### مکمل طور پر فائن ٹیوننگ مخصوص پرتیں۔

اڈاپٹر کے ساتھ اضافی ماڈیولز کو تربیت دینے کے لیے (مثال کے طور پر، زبان ماڈل ہیڈ)، ان کی وضاحت `modules_to_save` میں کریں۔ `modules_to_save` ان تہوں کو متعین کرتا ہے جو اڈاپٹر کے ساتھ مکمل طور پر ٹھیک ہیں، اس لیے *تمام* ان کے پیرامیٹرز کو اپ ڈیٹ کیا جاتا ہے۔ یہ اس وقت مفید ہوتا ہے جب کچھ تہوں کو اپ ڈیٹ کی ضرورت ہوتی ہے، مثال کے طور پر زبان ماڈل head (`lm_head`)، جب ترتیب کی درجہ بندی کے لیے causal LM کو اپناتے ہو۔

```py
lora_config = LoraConfig(
    modules_to_save=["lm_head"],
    ...
)
model.add_adapter(lora_config)
```

### منتخب کرنا کہ کن پرتوں کو اپنانا ہے۔

عام فن تعمیرات (Llama, Gemma, Qwen، وغیرہ) کے لیے، PEFT نے پہلے سے طے شدہ اہداف (جیسے `q_proj` اور `v_proj`)، اس لیے آپ کو `target_modules` بتانے کی ضرورت نہیں ہے۔ اگر آپ مختلف تہوں کو نشانہ بنانا چاہتے ہیں، یا ماڈل میں پہلے سے طے شدہ اہداف نہیں ہیں، تو `target_modules` کو ماڈیول کے ناموں کی فہرست یا ریجیکس پیٹرن کے طور پر واضح طور پر پاس کریں۔

```py
lora_config = LoraConfig(
    target_modules=["q_proj", "k_proj"],
    ...
)
model.add_adapter(lora_config)
```

## تربیت

ماڈل کو منسلک اڈاپٹر کے ساتھ [`Trainer`] پر پاس کریں اور [`~Trainer.train`] پر کال کریں۔ [`Trainer`] صرف اڈاپٹر کے پیرامیٹرز کو اپ ڈیٹ کرتا ہے (جو `requires_grad=True` والے ہیں) کیونکہ بیس ماڈل منجمد ہے۔

```py
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
```

ٹریننگ کے دوران، [`Trainer`] چیک پوائنٹس میں صرف اڈاپٹر کا وزن (`adapter_model.safetensors`) اور کنفیگریشن (`adapter_config.json`) ہوتا ہے، چیک پوائنٹس کو چھوٹا رکھتا ہے۔ بیس ماڈل شامل نہیں ہے۔

تربیت کے بعد، فائنل اڈاپٹر کو [`~PreTrainedModel.save_pretrained`] کے ساتھ محفوظ کریں۔

```py
model.save_pretrained("./my_adapter")
```

### چیک پوائنٹ سے دوبارہ شروع ہو رہا ہے

دوبارہ شروع کرنے پر [`Trainer`] خود بخود اڈاپٹر چیک پوائنٹس کا پتہ لگاتا ہے۔ [`Trainer`] چیک پوائنٹ ڈائرکٹری کو اسکین کرتا ہے جس میں اڈاپٹر کا وزن ہوتا ہے اور ہر اڈاپٹر کو درست قابل تربیت حالت کے ساتھ دوبارہ لوڈ کرتا ہے۔

```py
trainer.train(resume_from_checkpoint="./output/checkpoint-1000")
```

### تقسیم تربیت

PEFT اڈاپٹر تقسیم شدہ تربیت کے ساتھ کام کرتے ہیں۔

ZeRO-3 کے لیے، PEFT ماڈل کے ساتھ چیک پوائنٹس بچت کرتے وقت [`Trainer`] `exclude_frozen_parameters=True` پاس کرتا ہے۔ منجمد بیس ماڈل وزن کو چھوڑ دیا گیا ہے۔ صرف قابل تربیت اڈاپٹر کے پیرامیٹرز کو محفوظ کیا جاتا ہے، جس سے چیک پوائنٹ سائز کم ہوتا ہے اور وقت کی بچت ہوتی ہے۔

FSDP کے لیے، [`Trainer`] FSDP آٹو ریپ پالیسی کو درست طریقے سے LoRA پرتوں کو سنبھالنے کے لیے اپ ڈیٹ کرتا ہے۔ QLoRA (کوانٹائزڈ بیس ماڈل + LoRA) کے لیے، [`Trainer`] مخلوط درستگی کی پالیسی کو کوانٹائزیشن اسٹوریج کی قسم سے مماثل بھی بناتا ہے۔

آپ PEFT اڈاپٹر کو ٹینسر کے متوازی بیس ماڈل پر لوڈ کر سکتے ہیں اور PEFT کو اڈاپٹر کے وزن کو کم کرنے کی اجازت دے سکتے ہیں۔ اڈاپٹر لوڈ کرنے کے لیے `peft >= 0.21.1` کی ضرورت ہوتی ہے۔ TP کے ساتھ تربیت کے لیے [Tensor parallelism](https://huggingface.co/docs/transformers/main/en/tensor_parallelism) دیکھیں۔

## اڈاپٹر لوڈ ہو رہا ہے۔

اڈاپٹر لوڈ کرنے کے لیے، حب ریپوزٹری یا لوکل ڈائرکٹری میں ایک `adapter_config.json` فائل اور اڈاپٹر کا وزن ہونا چاہیے۔

<hfoptions id="load">
<hfoption id="from_pretrained">

[`~PreTrainedModel.from_pretrained`] خود بخود اڈیپٹرز کا پتہ لگاتا ہے۔ جب اسے `adapter_config.json` مل جاتا ہے، تو یہ صحیح بنیاد ماڈل لوڈ کرنے کے لیے `base_model_name_or_path` فیلڈ کو پڑھتا ہے، پھر اڈاپٹر کو اوپر لوڈ کرتا ہے۔

```py
from transformers import AutoModelForCausalLM

# Automatically loads the base model and attaches the adapter
model = AutoModelForCausalLM.from_pretrained("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
<hfoption id="load_adapter">

موجودہ ماڈل پر اڈاپٹر لوڈ کرنے کے لیے، [`~integrations.PeftAdapterMixin.load_adapter`] استعمال کریں۔

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-7b")
model.load_adapter("klcsp/gemma7b-lora-alpaca-11-v1")
```

</hfoption>
</hfoptions>

بڑے ماڈلز کے لیے، میموری کو بچانے کے لیے [bitsandbytes](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes) کے ساتھ 8-bit یا 4-bit precision میں کوانٹائزڈ ورژن لوڈ کریں۔ ماڈل کو دستیاب ہارڈ ویئر میں تقسیم کرنے کے لیے `device_map="auto"` شامل کریں۔

```py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    "klcsp/gemma7b-lora-alpaca-11-v1",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
)
```

## ایک سے زیادہ اڈاپٹر کا انتظام

A ماڈل ایک ہی وقت میں متعدد اڈاپٹر رکھ سکتا ہے۔ منفرد ناموں کے ساتھ اڈاپٹر شامل کریں، اور ضرورت کے مطابق ان کے درمیان سوئچ کریں۔

```py
from peft import LoraConfig

model.add_adapter(LoraConfig(r=8, lora_alpha=32), adapter_name="adapter_1")
model.add_adapter(LoraConfig(r=16, lora_alpha=64), adapter_name="adapter_2")
```

مخصوص اڈاپٹر کو چالو کرنے کے لیے [`~integrations.PeftAdapterMixin.set_adapter`] استعمال کریں۔ دوسرے اڈاپٹر غیر فعال ہیں لیکن میموری میں رہتے ہیں۔

```py
model.set_adapter("adapter_2")
```

[`~integrations.PeftAdapterMixin.enable_adapters`] تمام منسلک اڈاپٹرز کو فعال کرتا ہے، اور [`~integrations.PeftAdapterMixin.disable_adapters`] ان سب کو غیر فعال کرتا ہے۔

```py
# Disable all adapters for base model inference
model.disable_adapters()

# Re-enable all adapters
model.enable_adapters()
```

یہ دیکھنے کے لیے [`~integrations.PeftAdapterMixin.active_adapters`] استعمال کریں کہ فی الحال کون سے اڈاپٹر فعال ہیں۔

```py
model.active_adapters()
# ["adapter_1"]
```

مفت میموری کے لیے [`~integrations.PeftAdapterMixin.delete_adapter`] کے ساتھ اڈیپٹرز کو ہٹا دیں جن کی آپ کو مزید ضرورت نہیں ہے۔

```py
model.delete_adapter("adapter_1")
```

## ہاٹ سویپنگ اڈاپٹر

ہر بار جب آپ درخواست پیش کرتے ہیں تو نیا اڈاپٹر لوڈ کرنا نئی میموری مختص کرتا ہے۔ اگر ماڈل کو `torch.compile` کے ساتھ مرتب کیا گیا ہے، تو ہر نیا اڈاپٹر دوبارہ کمپائلیشن کو متحرک کرتا ہے۔ ہاٹ سویپنگ اڈاپٹر کے وزن کی جگہ لے لیتا ہے، دونوں مسائل سے گریز کرتا ہے۔ صرف LoRA اڈاپٹر تعاون یافتہ ہیں۔

LoRA اڈاپٹر لوڈ کرتے وقت `hotswap=True` پاس کریں تاکہ اس کے وزن کو موجودہ اڈاپٹر سلاٹ میں تبدیل کیا جا سکے۔ تبدیل کرنے کے لیے اڈاپٹر کے نام پر `adapter_name` سیٹ کریں (`"default"` پہلے سے طے شدہ اڈاپٹر کا نام ہے)۔

```py
model = AutoModel.from_pretrained(...)
# Load the first adapter normally
model.load_adapter(adapter_path_1)
# Generate outputs with adapter 1
...
# Hotswap the second adapter in-place
model.load_adapter(adapter_path_2, hotswap=True, adapter_name="default")
# Generate outputs with adapter 2
```

### torch.compile

مرتب شدہ ماڈلز کے لیے، [`~integrations.peft.PeftAdapterMixin.enable_peft_hotswap`] *پہلے اڈاپٹر کو لوڈ کرنے سے پہلے اور مرتب کرنے سے پہلے کال کریں۔

```py
model = AutoModel.from_pretrained(...)
max_rank = ...  # highest rank among all LoRAs you'll load
model.enable_peft_hotswap(target_rank=max_rank)
model.load_adapter(adapter_path_1, adapter_name="default")
model = torch.compile(model, ...)
output_1 = model(...)

# Hotswap without recompilation
model.load_adapter(adapter_path_2, adapter_name="default")
output_2 = model(...)
```

`target_rank` دلیل تمام LoRA اڈاپٹرز کے درمیان زیادہ سے زیادہ درجہ کا تعین کرتی ہے جو آپ لوڈ کریں گے۔ اگر آپ کے پاس رینک 8 اور رینک 16 والے اڈاپٹر ہیں تو `target_rank=16` پاس کریں۔ ڈیفالٹ 128 ہے۔

`enable_peft_hotswap` کال کرنے کے بعد، تمام بعد میں آنے والی `load_adapter` کالز ہاٹ سویپ بذریعہ ڈیفالٹ۔ ہاٹ سویپنگ کو غیر فعال کرنے کے لیے واضح طور پر `hotswap=False` پاس کریں۔

اگر ہاٹ سویپ شدہ اڈاپٹر ابتدائی اڈاپٹر سے زیادہ تہوں کو نشانہ بناتا ہے تو دوبارہ کمپائلیشن ہو سکتا ہے۔ اڈاپٹر کو لوڈ کریں جو سب سے زیادہ تہوں کو پہلے نشانہ بناتا ہے تاکہ دوبارہ مرتب ہونے سے بچ سکے۔

> [!TIP] > غیر متوقع دوبارہ ترتیب کا پتہ لگانے کے لیے اپنے کوڈ کو `with torch._dynamo.config.patch(error_on_recompile=True)` میں لپیٹیں۔ اگر آپ مندرجہ بالا مراحل پر عمل کرنے کے باوجود دوبارہ کمپائلیشن کا پتہ لگاتے ہیں، تو دوبارہ پیدا کرنے والی مثال کے ساتھ [PEFT](https://github.com/huggingface/peft/issues) کے ساتھ ایک مسئلہ کھولیں۔

## اگلے اقدامات

- PEFT [دستاویزات](https://huggingface.co/docs/peft/index) PEFT طریقوں اور اختیارات کی پوری رینج کا احاطہ کرتا ہے۔
- PEFT [ہاٹ سویپنگ ریفرنس](https://huggingface.co/docs/peft/main/en/package_reference/hotswap#peft.utils.hotswap.hotswap_adapter) حدود اور ایج کیسز کی تفصیلات۔
- A [بلاگ پوسٹ](https://huggingface.co/blog/lora-fast) بینچ مارکس `torch.compile` ہاٹ سویپنگ کے ساتھ رن ٹائم کو کیسے بہتر بناتا ہے۔
