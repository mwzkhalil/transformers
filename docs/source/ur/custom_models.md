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

# حسب ضرورت ماڈلز

Transformers ماڈلزمرضی کے مطابق ہونے کے لیے ڈیزائن کیے گئے ہیں۔ اےماڈلزکوڈ مکمل طور پر موجود ہے [ماڈل](https://github.com/huggingface/transformers/tree/main/src/transformers/models)کا ذیلی فولڈرTransformersذخیرہ ہر فولڈر میں a`modeling.py`اور a`configuration.py`فائل اپنی مرضی کے مطابق بنانا شروع کرنے کے لیے ان فائلوں کو کاپی کریں۔ماڈل.

> [!TIP] > اگر آپ بالکل نیا ماڈل بنا رہے ہیں تو شروع سے شروع کرنا آسان ہو سکتا ہے۔ لیکن ماڈلز کے لیے جو Transformers میں موجود ایک سے بہت ملتے جلتے ہیں، اسی ترتیب اور ماڈل کلاس کو دوبارہ استعمال یا ذیلی کلاس کرنا تیز تر ہے۔

یہ گائیڈ آپ کو دکھائے گا کہ کس طرح ResNet ماڈل کو اپنی مرضی کے مطابق بنانا ہے، [AutoClass](#autoclass) سپورٹ کو فعال کرنا ہے، اور اسے حب پر شیئر کرنا ہے۔

## کنفیگریشن

بیس [`PreTrainedConfig`] کلاس کے ذریعے دی گئی ایک ترتیب، ماڈل بنانے کے لیے تمام ضروری معلومات پر مشتمل ہے۔ یہ وہ جگہ ہے جہاں آپ حسب ضرورت ResNet ماڈل کی خصوصیات کو ترتیب دیں گے۔ مختلف صفات مختلف ResNet ماڈل اقسام دیتی ہیں۔

کنفیگریشن کو اپنی مرضی کے مطابق کرنے کے بنیادی اصول یہ ہیں:

1. حسب ضرورت کنفیگریشن [`PreTrainedConfig`] ذیلی کلاس ہونی چاہیے۔ یہ یقینی بناتا ہے کہ ایک حسب ضرورت ماڈل میں Transformers' ماڈل جیسے [`~PreTrainedConfig.from_pretrained`]، [`~PreTrainedConfig.save_pretrained`]، اور [`~PreTrainedConfig.push_to_hub`] کی تمام فعالیت موجود ہے۔
2. [`PreTrainedConfig`] `__init__` کو کسی بھی `kwargs` کو قبول کرنا چاہیے اور انہیں سپر کلاس `__init__` میں منتقل کرنا چاہیے۔ [`PreTrainedConfig`] میں آپ کی حسب ضرورت کنفیگریشن میں سیٹ کردہ فیلڈز سے زیادہ فیلڈز ہیں، اس لیے جب آپ [`~PreTrainedConfig.from_pretrained`] کے ساتھ کنفیگریشن لوڈ کرتے ہیں، تو ان فیلڈز کو آپ کی کنفیگریشن کے ذریعے قبول کرنے اور سپر کلاس میں منتقل کرنے کی ضرورت ہوتی ہے۔

> [!TIP] > کچھ پیرامیٹرز کی درستگی کی جانچ کرنا مفید ہے۔ ذیل کی مثال میں، `block_type` اور `stem_type` پہلے سے طے شدہ قدروں میں سے ایک سے تعلق کو یقینی بنانے کے لیے ایک چیک لاگو کیا گیا ہے۔ [AutoClass](#autoclass) سپورٹ کو فعال کرنے کے لیے کنفیگریشن کلاس میں `model_type` شامل کریں۔

```py
from transformers import PreTrainedConfig
from typing import List

class ResnetConfig(PreTrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers: list[int] = [3, 4, 6, 3],
        num_classes: int = 1000,
        input_channels: int = 3,
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: str = "",
        avg_down: bool = False,
        **kwargs,
    ):
        if block_type not in ["basic", "bottleneck"]:
            raise ValueError(f"`block_type` must be 'basic' or bottleneck', got {block_type}.")
        if stem_type not in ["", "deep", "deep-tiered"]:
            raise ValueError(f"`stem_type` must be '', 'deep' or 'deep-tiered', got {stem_type}.")

        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down
        super().__init__(**kwargs)
```

کنفیگریشن کو JSON فائل میں اپنے حسب ضرورت ماڈل فولڈر، `custom-resnet`، [`~PreTrainedConfig.save_pretrained`] کے ساتھ محفوظ کریں۔

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d_config.save_pretrained("custom-resnet")
```

## ماڈل

حسب ضرورت ResNet کنفیگریشن کے ساتھ، اب آپ ماڈل کو تخلیق اور اپنی مرضی کے مطابق بنا سکتے ہیں۔ ماڈل بیس [`PreTrainedModel`] کلاس کو ذیلی کلاس کرتا ہے۔ [`PreTrainedConfig`] کی طرح، [`PreTrainedModel`] سے وراثت میں ملنا اور کنفیگریشن کے ساتھ سپر کلاس کو شروع کرنا Transformers' فنکشنلٹیز کو بڑھاتا ہے جیسے محفوظ کرنا اور اپنی مرضی کے مطابق ماڈل پر لوڈ کرنا۔

Transformers' ماڈلز `__init__` طریقہ میں `config` آبجیکٹ کو قبول کرنے کے کنونشن کی پیروی کریں۔ یہ `config` آبجیکٹ کو متعدد دلائل میں توڑنے کے بجائے پورے `config` کو ماڈل ذیلی پرتوں میں منتقل کرتا ہے جو انفرادی طور پر ذیلی سطحوں کو بھیجے جاتے ہیں۔

ماڈلز اس طرح لکھنا کسی بھی ہائپر پیرامیٹر کے لیے سچائی کے واضح ذریعہ کے ساتھ آسان کوڈ تیار کرتا ہے۔ یہ دوسرے Transformers' ماڈلز سے کوڈ کو دوبارہ استعمال کرنا بھی آسان بناتا ہے۔

آپ دو ResNet ماڈلز بنائیں گے، ایک barebones ResNet ماڈل جو چھپی ہوئی حالتوں کو آؤٹ پٹ کرتا ہے اور ایک ResNet ماڈل تصویر کی درجہ بندی کے سر کے ساتھ۔

<hfoptions id="resnet">
<hfoption id="ResnetModel">

بلاک کی اقسام اور کلاسوں کے درمیان میپنگ کی وضاحت کریں۔ باقی سب کچھ کنفیگریشن کلاس کو ResNet ماڈل کلاس میں دے کر تخلیق کیا جاتا ہے۔

> [!TIP] > [AutoClass](#autoclass) سپورٹ کو فعال کرنے کے لیے `config_class` کو ماڈل کلاس میں شامل کریں۔

```py
from transformers import PreTrainedModel
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from .configuration_resnet import ResnetConfig

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}

class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor):
        return self.model.forward_features(tensor)
```

</hfoption>
<hfoption id="ResnetModelForImageClassification">

`forward` طریقہ کو دوبارہ لکھنے کی ضرورت ہے کہ اگر لیبل دستیاب ہوں تو ہر لاگٹ کے نقصان کا حساب لگایا جائے۔ بصورت دیگر، ResNet ماڈل کلاس ایک جیسی ہے۔

> [!TIP] > [AutoClass](#autoclass) سپورٹ کو فعال کرنے کے لیے `config_class` کو ماڈل کلاس میں شامل کریں۔

```py
import torch

class ResnetModelForImageClassification(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
```

</hfoption>
</hfoptions>

A ماڈل کسی بھی آؤٹ پٹ فارمیٹ کو واپس کر سکتا ہے۔ لیبل دستیاب ہونے پر نقصانات کے ساتھ لغت (جیسے `ResnetModelForImageClassification`) واپس کرنا حسب ضرورت ماڈل کو [`Trainer`] کے ساتھ ہم آہنگ بنا دیتا ہے۔ دوسرے آؤٹ پٹ فارمیٹس کے لیے، آپ کو تربیت کے لیے اپنے ٹریننگ لوپ یا ایک مختلف لائبریری کی ضرورت ہوگی۔

کنفیگریشن کے ساتھ حسب ضرورت ماڈل کلاس کو فوری بنائیں۔

```py
resnet50d = ResnetModelForImageClassification(resnet50d_config)
```

اس مقام پر، آپ پہلے سے تربیت یافتہ وزن ماڈل میں لوڈ کر سکتے ہیں یا اسے شروع سے تربیت دے سکتے ہیں۔ اس گائیڈ میں، آپ پہلے سے تربیت یافتہ وزن لوڈ کریں گے۔

پہلے سے تربیت یافتہ وزن کو [timm](https://hf.co/docs/timm/index) لائبریری سے لوڈ کریں، اور پھر ان وزنوں کو اپنی مرضی کے مطابق ماڈل میں [load_state_dict](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict) کے ساتھ منتقل کریں۔

```py
import timm

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

## AutoClass

[AutoClass](https://huggingface.co/docs/transformers/main/en/models#model-classes) API ایک دیے گئے ماڈل کے لیے درست فن تعمیر کو خودکار طور پر لوڈ کرنے کے لیے ایک شارٹ کٹ ہے۔ آپ کے حسب ضرورت ماڈل لوڈ کرنے والے صارفین کے لیے اسے فعال کرنا آسان ہے۔

یقینی بنائیں کہ آپ کے پاس کنفیگریشن کلاس میں `model_type` وصف (موجودہ ماڈل اقسام سے مختلف ہونا چاہیے) اور ماڈل کلاس میں `config_class` وصف موجود ہے۔ حسب ضرورت کنفیگریشن اور ماڈل کو [AutoClass](https://huggingface.co/docs/transformers/main/en/models#model-classes) API میں شامل کرنے کے لیے [`~AutoConfig.register`] طریقہ استعمال کریں۔

> [!ٹپ] > پہلی دلیل[`AutoConfig.register`]سے ملنا چاہئے`model_type`کسٹم کنفیگریشن کلاس میں انتساب، اور پہلی دلیل[`AutoModel.register`]سے ملنا چاہئے`config_class`رواج کےماڈلکلاس

```py
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification

AutoConfig.register("resnet", ResnetConfig)
AutoModel.register(ResnetConfig, ResnetModel)
AutoModelForImageClassification.register(ResnetConfig, ResnetModelForImageClassification)
```

آپ کا حسب ضرورت ماڈل کوڈ اب [AutoClass](#autoclass) API کے ساتھ مطابقت رکھتا ہے۔ صارفین ماڈل کو [AutoModel](https://huggingface.co/docs/transformers/main/en/model_doc/auto) یا [`AutoModelForImageClassification`] کلاسز کے ساتھ لوڈ کر سکتے ہیں۔

## اپ لوڈ کریں۔

ایک حسب ضرورت ماڈل [Hub](https://hf.co/models) پر اپ لوڈ کریں تاکہ دوسرے صارفین اسے آسانی سے لوڈ اور استعمال کرسکیں۔

یقینی بنائیں کہ ماڈل ڈائرکٹری صحیح طریقے سے ترتیب دی گئی ہے جیسا کہ ذیل میں دکھایا گیا ہے۔ ڈائریکٹری میں شامل ہونا چاہئے:

- `modeling.py`: `ResnetModel` اور `ResnetModelForImageClassification` کے لیے کوڈ پر مشتمل ہے۔ یہ فائل دوسری فائلوں کے لیے متعلقہ درآمدات پر انحصار کر سکتی ہے جب تک کہ وہ ایک ہی ڈائریکٹری میں ہوں۔

> [!انتباہ] > Transformers' ماڈل فائل کاپی کرتے وقت، تمام متعلقہ درآمدات کو `modeling.py` فائل کے اوپری حصے میں تبدیل کریں تاکہ Transformers سے درآمد کریں۔

- `configuration.py`: `ResnetConfig` کے لیے کوڈ پر مشتمل ہے۔
- `__init__.py`: خالی ہوسکتی ہے، یہ فائل Python `resnet_model` کو ماڈیول کے طور پر استعمال کرنے کی اجازت دیتی ہے۔

```bash
.
└── resnet_model
    ├── __init__.py
    ├── configuration_resnet.py
    └── modeling_resnet.py
```

ماڈل شیئر کرنے کے لیے، ResNet ماڈل اور کنفیگریشن درآمد کریں۔

```py
from resnet_model.configuration_resnet import ResnetConfig
from resnet_model.modeling_resnet import ResnetModel, ResnetModelForImageClassification
```

کوڈ کو ماڈل اور کنفیگریشن فائلوں سے کاپی کریں۔ AutoClass اشیاء کو [`~PreTrainedModel.save_pretrained`] کے ساتھ محفوظ کرنے کے لیے، [`~PreTrainedConfig.register_for_auto_class`] طریقہ کو کال کریں۔ یہ AutoClass آبجیکٹ اور میپنگ کو شامل کرنے کے لیے JSON فائل کی ترتیب میں ترمیم کرتا ہے۔

ماڈل کے لیے، کام کی بنیاد پر مناسب `AutoModelFor` کلاس چنیں۔

```py
ResnetConfig.register_for_auto_class()
ResnetModel.register_for_auto_class("AutoModel")
ResnetModelForImageClassification.register_for_auto_class("AutoModelForImageClassification")
```

ماڈل میں ایک سے زیادہ کاموں کا نقشہ بنانے کے لیے، JSON فائل میں براہ راست `auto_map` میں ترمیم کریں۔

```json
"auto_map": {
    "AutoConfig": "<your-repo-name>--<config-name>",
    "AutoModel": "<your-repo-name>--<config-name>",
    "AutoModelFor<Task>": "<your-repo-name>--<config-name>",    
},
```

کنفیگریشن بنائیں اور ماڈل اور اس میں پہلے سے تربیت یافتہ وزن لوڈ کریں۔

```py
resnet50d_config = ResnetConfig(block_type="bottleneck", stem_width=32, stem_type="deep", avg_down=True)
resnet50d = ResnetModelForImageClassification(resnet50d_config)

pretrained_model = timm.create_model("resnet50d", pretrained=True)
resnet50d.model.load_state_dict(pretrained_model.state_dict())
```

ماڈل اب حب میں بھیجے جانے کے لیے تیار ہے۔ کمانڈ لائن یا نوٹ بک سے اپنے Hugging Face اکاؤنٹ میں لاگ ان کریں۔

<hfoptions id="push">
<hfoption id="huggingface-CLI">

```bash
hf auth login
```

</hfoption>
<hfoption id="notebook">

```py
from huggingface_hub import notebook_login

notebook_login()
```

</hfoption>
</hfoptions>

ماڈل کو حب پر اپ لوڈ کرنے کے لیے ماڈل پر [`~PreTrainedModel.push_to_hub`] کو کال کریں۔

```py
resnet50d.push_to_hub("custom-resnet50d")
```

پہلے سے تربیت یافتہ وزن، ترتیب، `modeling.py` اور `configuration.py` فائلوں کو اب آپ کے نام کی جگہ کے نیچے [repository](https://hf.co/sgugger/custom-resnet50d) میں حب پر اپ لوڈ کیا جانا چاہیے۔

چونکہ ایک حسب ضرورت ماڈل Transformers' ماڈل جیسا ماڈلنگ کوڈ استعمال نہیں کرتا ہے، اس لیے آپ کو اسے لوڈ کرنے کے لیے `trust_remote_code=True` کو [`~PreTrainedModel.from_pretrained`] میں شامل کرنا ہوگا۔ مزید معلومات کے لیے لوڈ [کسٹم ماڈلز](https://huggingface.co/docs/transformers/main/en/models#custom-models) سیکشن سے رجوع کریں۔
