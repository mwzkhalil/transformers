<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ایک GPU ورک سٹیشن بنانا

گہری سیکھنے کی مشین بناتے وقت GPU سب سے اہم انتخاب میں سے ایک ہے۔ ٹینسر کور میٹرکس ضرب کو مؤثر طریقے سے ہینڈل کرتے ہیں، اور ہائی میموری بینڈوڈتھ ڈیٹا کو رواں دواں رکھتی ہے۔ بڑی ماڈلز تربیت کے لیے زیادہ طاقتور GPU، ایک سے زیادہ GPUs، یا آف لوڈنگ تکنیک کی ضرورت ہوتی ہے جو کام کو CPU یا NVMe میں منتقل کرتی ہے۔

نیچے دی گئی تجاویز گہری سیکھنے کے لیے عملی GPU سیٹ اپ کا احاطہ کرتی ہیں۔

## طاقت

اعلی درجے کے صارفین کے GPU میں دو یا تین PCIe 8 پن پاور ساکٹ ہو سکتے ہیں۔ ایک علیحدہ 12V PCIe 8 پن کیبل ہر ساکٹ سے جوڑیں۔ دو ساکٹ کو جوڑنے کے لیے *پگ ٹیل کیبل* (ایک ہی کیبل جس کے ایک سرے پر دو اسپلٹ ہیں) استعمال نہ کریں، بصورت دیگر، آپ کو GPU سے مکمل کارکردگی نہیں ملے گی۔

ہر PCIe 8 پن پاور کیبل کو پاور سپلائی یونٹ (PSU) پر 12V ریل سے جوڑیں۔ ہر کیبل 150W تک فراہم کرتی ہے۔ کچھ GPUs PCIe 12-pin کنیکٹر استعمال کرتے ہیں جو 500-600W تک فراہم کرتا ہے۔ لوئر اینڈ GPUs ایک PCIe 6 پن کنیکٹر استعمال کر سکتے ہیں جو 75W تک فراہم کرتا ہے۔

ایک PSU کو مستحکم وولٹیج کو برقرار رکھنا چاہیے کیونکہ غیر مستحکم وولٹیج چوٹی کے استعمال کے دوران پاور کے GPU کو بھوکا رکھ سکتا ہے۔

## کولنگ

زیادہ گرم GPU کارکردگی کو تھروٹل کرتا ہے اور نقصان کو روکنے کے لیے بند کر دیتا ہے۔ مکمل کارکردگی اور طویل عمر کے لیے درجہ حرارت 158–167°F (70–75 سیلسیس) کے درمیان رکھیں۔ 183–194°F (84–90 سیلسیس) سے اوپر، GPU عام طور پر تھروٹلنگ شروع کر دیتا ہے۔

## ملٹی جی پی یو کنیکٹوٹی

آپ کے GPUs ملٹی GPU سیٹ اپس کے لیے معاملات کو کیسے جوڑتے ہیں۔ [NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) کنکشنز PCIe پلوں سے زیادہ تیز ہیں، لیکن اثر آپ کی متوازی حکمت عملی پر منحصر ہے۔ DDP میں زیرو سے کم GPU سے GPU مواصلات ہے، لہذا کنکشن کی رفتار کم اہمیت رکھتی ہے۔

آپ کے GPUs کیسے جڑے ہوئے ہیں یہ چیک کرنے کے لیے نیچے دی گئی کمانڈ کو چلائیں۔

```bash
nvidia-smi topo -m
```

<hfoptions id="nvlink">
<hfoption id="NVLink">

[NVLink](https://www.nvidia.com/en-us/design-visualization/nvlink-bridges/) متعدد GPUs کو جوڑنے کے لیے NVIDIA کا تیز رفتار مواصلاتی نظام ہے۔

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NV2     0-23            N/A
GPU1    NV2      X      0-23            N/A
```

`NV2` اشارہ کرتا ہے `GPU0` اور `GPU1` 2 NVLinks کے ذریعے منسلک ہیں۔

</hfoption>
<hfoption id="PCIe bridge">

```bash
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      PHB     0-11            N/A
GPU1    PHB      X      0-11            N/A
```

`PHB` اشارہ کرتا ہے `GPU0` اور `GPU1` ایک PCIe پل کے ذریعے جڑے ہوئے ہیں۔

</hfoption>
</hfoptions>

## اگلے اقدامات

-  GPUs کے گہرے موازنہ کے لیے [کون سا GPU(s) حاصل کرنا ہے ڈیپ لرننگ](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/) بلاگ پوسٹ دیکھیں۔
