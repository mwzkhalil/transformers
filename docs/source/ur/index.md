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

# Transformers

<h3 align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/transformers_as_a_model_definition.png"/>
</h3>

Transformers متن، کمپیوٹر وژن، آڈیو، ویڈیو اور ملٹی موڈل ماڈلز کے لیے جدید مشین لرننگ ماڈلز کی تعریف کا فریم ورک ہے۔ اسے انفرنس (inference) اور تربیت، دونوں کے لیے استعمال کیا جاتا ہے۔

یہ ماڈل کی تعریف کو ایک جگہ رکھتا ہے تاکہ پورے ماحولیاتی نظام میں اس تعریف پر اتفاق ہو۔ `transformers` مختلف فریم ورکس کے درمیان رابطے کا مرکزی ذریعہ ہے: اگر کسی ماڈل کی تعریف اس میں دستیاب ہو تو وہ زیادہ تر تربیتی فریم ورکس (Axolotl، Unsloth، DeepSpeed، FSDP، PyTorch-Lightning، ...) اور انفرنس انجنز (vLLM، SGLang، TGI، ...) کے ساتھ، نیز ماڈل کی تعریف استعمال کرنے والی متعلقہ لائبریریوں (llama.cpp، mlx، ...) کے ساتھ مطابقت رکھے گی۔

ہم نئے جدید ترین ماڈلز کی معاونت کرنے اور ان کے استعمال کو زیادہ لوگوں تک پہنچانے کے لیے کوشاں ہیں، تاکہ ان ماڈلز کی تعریف سادہ، حسبِ ضرورت قابلِ تبدیلی اور مؤثر ہو۔

[Hugging Face Hub](https://huggingface.com/models) پر Transformers کے [ماڈل چیک پوائنٹس](https://huggingface.co/models?library=transformers&sort=trending) کی تعداد 1M+ سے زیادہ ہے، جنہیں آپ استعمال کر سکتے ہیں۔

آج ہی [Hub](https://huggingface.com/) دیکھیں، کوئی ماڈل تلاش کریں اور Transformers کے ساتھ فوراً کام شروع کریں۔

Transformers میں متن، وژن، آڈیو اور ملٹی موڈل ماڈلز کے جدید ترین آرکیٹیکچرز جاننے کے لیے [Models Timeline](https://huggingface.co/docs/transformers/main/en/models_timeline) دیکھیں۔

## خصوصیات

Transformers پہلے سے تربیت یافتہ جدید ترین ماڈلز کے ساتھ انفرنس یا تربیت کے لیے درکار سہولتیں فراہم کرتا ہے۔ چند اہم خصوصیات یہ ہیں:

- [Pipeline](https://huggingface.co/docs/transformers/main/en/pipeline_tutorial): متن بنانے، تصویر کو حصوں میں تقسیم کرنے، خودکار گفتار کی شناخت، دستاویز سے سوالوں کے جواب دینے اور مشین لرننگ کے کئی دوسرے کاموں کے لیے ایک سادہ اور بہتر انفرنس کلاس۔
- [Trainer](https://huggingface.co/docs/transformers/main/en/trainer): ایک جامع ٹرینر جو PyTorch ماڈلز کی تربیت اور تقسیم شدہ تربیت کے لیے mixed precision، torch.compile اور FlashAttention جیسی سہولتیں فراہم کرتا ہے۔
- [generate](https://huggingface.co/docs/transformers/main/en/llm_tutorial): بڑے زبان ماڈلز (LLMs) اور وژن لینگویج ماڈلز (VLMs) سے تیز رفتار متن بنانے کے لیے، جس میں streaming اور کئی decoding حکمتِ عملیوں کی معاونت شامل ہے۔

## ڈیزائن

> [!TIP]
> Transformers کے ڈیزائن کے اصولوں کے بارے میں مزید جاننے کے لیے اس کا [فلسفہ](https://huggingface.co/docs/transformers/main/en/philosophy) پڑھیں۔

Transformers ڈویلپرز، مشین لرننگ انجینیئرز اور محققین کے لیے بنایا گیا ہے۔ اس کے بنیادی ڈیزائن اصول یہ ہیں:

1. تیز اور آسان استعمال: ہر ماڈل صرف تین بنیادی کلاسوں (configuration، model اور preprocessor) سے نافذ کیا جاتا ہے، اور [`Pipeline`] یا [`Trainer`] کے ذریعے اسے انفرنس یا تربیت کے لیے جلد استعمال کیا جا سکتا ہے۔
2. پہلے سے تربیت یافتہ ماڈلز: بالکل نیا ماڈل تربیت دینے کے بجائے پہلے سے تربیت یافتہ ماڈل استعمال کرکے کاربن کے اخراج، کمپیوٹ کی لاگت اور وقت کو کم کریں۔ ہر ماڈل کو حتی الامکان اپنے اصل ماڈل کے مطابق دوبارہ تیار کیا جاتا ہے اور وہ جدید ترین کارکردگی فراہم کرتا ہے۔

<div class="flex justify-center">
  <a target="_blank" href="https://huggingface.co/support">
      <img alt="HuggingFace Expert Acceleration Program" src="https://hf.co/datasets/huggingface/documentation-images/resolve/81d7d9201fd4ceb537fc4cebc22c29c37a2ed216/transformers/transformers-index.png" style="width: 100%; max-width: 600px; border: 1px solid #eee; border-radius: 4px; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);">
  </a>
</div>

## سیکھیں

اگر آپ Transformers میں نئے ہیں یا ٹرانسفارمر ماڈلز کے بارے میں مزید سیکھنا چاہتے ہیں تو ہم [LLM course](https://huggingface.co/learn/llm-course/chapter1/1?fw=pt) سے آغاز کرنے کی تجویز دیتے ہیں۔ یہ جامع کورس ٹرانسفارمر ماڈلز کے بنیادی طریقۂ کار سے لے کر مختلف کاموں میں ان کے عملی استعمال تک سب کچھ سکھاتا ہے۔ اس میں آپ مکمل عملی طریقۂ کار سیکھیں گے: اعلیٰ معیار کے ڈیٹاسیٹس تیار کرنا، بڑے زبان ماڈلز کی فائن ٹیوننگ (fine-tuning) کرنا، اور ان میں استدلال کی صلاحیت پیدا کرنا۔ بنیادی تصورات سیکھنے کے ساتھ ساتھ نظری اور عملی مشقیں بھی آپ کو ان کی مضبوط سمجھ بنانے میں مدد دیں گی۔

آگے بڑھنے کے لیے [تنصیب](./installation) اور [فوری آغاز](./quicktour) دیکھیں۔
