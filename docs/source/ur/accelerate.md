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

# Accelerate

[Accelerate](https://hf.co/docs/accelerate/index) تقسیم شدہ تربیتی بیک اینڈز جیسے [FSDP](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) یا [DeepSpeed](https://www.deepspeed.ai/) کے لیے ایک متحد انٹرفیس فراہم کرتا ہے۔ یہ آپ کے ماحول کا پتہ لگاتا ہے (GPUs کی تعداد، تقسیم شدہ بیک اینڈ، مخلوط درستگی وغیرہ) اور خود بخود تربیت کو ترتیب دیتا ہے، چاہے آپ DDP کے ساتھ 1 GPU یا FSDP کے ساتھ 8 GPU پر ہوں۔

Accelerate ماڈل کو مناسب تقسیم شدہ ریپر میں لپیٹتا ہے، اسے صحیح ڈیوائس پر لے جاتا ہے، اور ایک مطابقت پذیر آپٹیمائزر بناتا ہے۔ ٹریننگ کے دوران، Accelerate مخلوط درستگی کے لیے گریڈینٹ اسکیلنگ کو سنبھالنے کے لیے اپنا [`~accelerate.Accelerator.backward`] طریقہ استعمال کرتا ہے۔ [`Trainer`] مناسب Accelerate APIs کو کال کرتا ہے اور تمام تقسیم شدہ میکانکس کو Accelerate پر بھیجتا ہے۔

Accelerate کنفگ فائل یا [`TrainingArguments`] کے ساتھ [`Trainer`] کے لیے Accelerate کو کنفیگر کریں۔

## Accelerate کی ترتیب فائل

[accelerate config](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-config) کمانڈ چلائیں اور اپنے ہارڈ ویئر اور ٹریننگ سیٹ اپ کے بارے میں سوالات کے جواب دیں۔ یہ آپ کے کیشے میں `default_config.yaml` فائل بناتا ہے۔ ذیل کی مثال FSDP کے لیے ہے۔

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_version: 2
  fsdp_reshard_after_forward: true
  fsdp_cpu_offload: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_activation_checkpointing: false
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
mixed_precision: bf16
num_machines: 1
num_processes: 4
```

[`Trainer`] پر مبنی اسکرپٹ کے ساتھ [accelerate launch](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) چلائیں، اور Accelerate ٹریننگ ترتیب دینے کے لیے کنفگ فائل کو پڑھتا ہے۔ [`~TrainingArguments#fsdp_config`] اور [`~TrainingArguments#deepspeed`] args غیر ضروری ہیں کیونکہ Accelerate config فائل انہی ترتیبات کا احاطہ کرتی ہے۔

```cli
accelerate launch train.py
```

[`~TrainingArguments#accelerator_config`] ان ترتیبات کو قبول کرتا ہے جن میں سرشار اعلیٰ سطحی دلائل نہیں ہوتے ہیں۔ مثال کے طور پر، اعلی GPU تھرو پٹ کے لیے کمپیوٹ کے ساتھ ڈیٹا کی منتقلی کو اوورلیپ کرنے کے لیے `non_blocking=True` کو [`~TrainingArguments.dataloader_pin_memory`] کے ساتھ سیٹ کریں۔

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    dataloader_pin_memory=True,
    accelerator_config={
        "non_blocking": True,
    },
)
```

## تربیتی دلائل

بیک اینڈ کے لیے مخصوص کنفیگریشن کو [`TrainingArguments`] پر منتقل کریں۔ [`~Trainer.create_accelerator_and_postprocess`] طریقہ ترتیبات کو پڑھتا ہے اور تربیت کو ترتیب دیتا ہے۔

<hfoptions id="backend">
<hfoption id="FSDP">

JSON کنفیگریشن فائل پاس کریں یا [`~TrainingArguments.fsdp_config`] پر ڈکٹ بھیجیں۔ مکمل گائیڈ اور کنفیگریشن ریفرنس کے لیے [FSDP](https://huggingface.co/docs/transformers/main/en/fsdp) دیکھیں۔

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    fsdp=True,
    fsdp_config="path/to/fsdp.json",
)
```

</hfoption>
<hfoption id="DeepSpeed">

JSON کنفیگریشن فائل پاس کریں یا [`~TrainingArguments.deepspeed`] پر ڈکٹ بھیجیں۔ مکمل گائیڈ اور تشکیل کے حوالے کے لیے [DeepSpeed](https://huggingface.co/docs/transformers/main/en/deepspeed) دیکھیں۔

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    deepspeed="path/to/ds_config.json",
)
```

</hfoption>
<hfoption id="DDP">

DDP براہ راست [`TrainingArguments`] فیلڈز کے ذریعے ترتیب دیا گیا ہے۔ تفصیلات کے لیے [DDP](https://huggingface.co/docs/transformers/main/en/ddp) دیکھیں۔

```py
from transformers import TrainingArguments

TrainingArguments(
    ...,
    ddp_backend="nccl",
    ddp_find_unused_parameters=False,
    ddp_bucket_cap_mb=25,
    ddp_timeout=1800,
)
```

</hfoption>
</hfoptions>

## اگلے اقدامات

- ڈیٹا متوازی تربیت کے لیے [DDP](https://huggingface.co/docs/transformers/main/en/ddp) دیکھیں جب آپ کا ماڈل ایک GPU پر فٹ ہوجاتا ہے۔
- GPUs میں شارڈنگ پیرامیٹرز، گریڈینٹ، اور آپٹیمائزر اسٹیٹس کے لیے [FSDP](https://huggingface.co/docs/transformers/main/en/fsdp) دیکھیں۔
- ZeRO آپٹیمائزیشن اور آف لوڈنگ کے لیے [DeepSpeed](https://huggingface.co/docs/transformers/main/en/deepspeed) دیکھیں۔
