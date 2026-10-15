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

# شیئرنگ

Hugging Face [Hub](https://hf.co/models) تمام مختلف اقسام اور سائز کے ماڈلز اشتراک کرنے، دریافت کرنے اور استعمال کرنے کا ایک پلیٹ فارم ہے۔ ہم سب کے لیے اوپن سورس مشین لرننگ کو آگے بڑھانے کے لیے اپنے ماڈل کو حب پر شیئر کرنے کی انتہائی سفارش کرتے ہیں!

یہ گائیڈ آپ کو دکھائے گا کہ ماڈل کو Transformers سے حب میں کیسے بانٹنا ہے۔

## سیٹ اپ

حب میں ماڈل کا اشتراک کرنے کے لیے، آپ کو ایک Hugging Face [اکاؤنٹ](https://hf.co/join) کی ضرورت ہے۔ ایک [User Access Token](https://hf.co/docs/hub/security-tokens#user-access-tokens) (بذریعہ ڈیفالٹ [cache](./installation#cache-directory) میں محفوظ) بنائیں اور کمانڈ لائن یا نوٹ بک سے اپنے اکاؤنٹ میں لاگ ان کریں۔

<hfoptions id="share">
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

## ذخیرہ کی خصوصیات

<Youtube id="XvSGPZFEjDY"/>

ہر ماڈل ریپوزٹری میں ورژننگ، کمٹ ہسٹری، اور ڈیف ویژولائزیشن کی خصوصیات ہیں۔

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/vis_diff.png"/>
</div>

ورژننگ [Git](https://git-scm.com/) اور [Git Large File Storage (LFS)](https://git-lfs.github.com/) پر مبنی ہے، اور یہ نظرثانی کو قابل بناتا ہے، کمٹ ہیش، ٹیگ یا برانچ کے ساتھ ماڈل ورژن کی وضاحت کرنے کا ایک طریقہ۔

مثال کے طور پر، کمٹ ہیش سے مخصوص ماڈل ورژن لوڈ کرنے کے لیے [`~PreTrainedModel.from_pretrained`] میں `revision` پیرامیٹر استعمال کریں۔

```py
model = AutoModel.from_pretrained(
    "julien-c/EsperBERTo-small", revision="4c77982"
)
```

برانچ یا ٹیگ سے لوڈ کرنا اس کال میں ہر فائل کو ایک کمٹ پر پن کرتا ہے، پھر اس کمٹ سے ہر فائل کو بازیافت کرتا ہے۔

وہ پن صرف کال کے لیے رہتا ہے۔ تمام رنز میں دوبارہ پیدا کرنے کے لیے، ایک واضح کمٹ ہیش یا ٹیگ پاس کریں جسے آپ فکسڈ سمجھتے ہیں۔ `main` یا کسی دوسری حرکت پذیر شاخ پر بھروسہ نہ کریں جو رنز کے درمیان فکس رہے۔

ماڈل ذخیرہ گاہیں [گیٹنگ](https://hf.co/docs/hub/models-gated) کو بھی سپورٹ کرتی ہیں تاکہ یہ کنٹرول کیا جا سکے کہ ماڈل تک کون رسائی حاصل کر سکتا ہے۔ گیٹنگ استعمال کرنے والوں کے منتخب گروپ کو کسی تحقیق کے عوامی ہونے سے پہلے ماڈل کا جائزہ لینے کی اجازت دینے کے لیے عام ہے۔

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/gated-model.png"/>
</div>

ایک ماڈل ذخیرہ میں ایک انفرنس [ویجیٹ](https://hf.co/docs/hub/models-widgets) بھی شامل ہوتا ہے تاکہ صارفین حب پر ماڈل کے ساتھ براہ راست تعامل کر سکیں۔

مزید معلومات کے لیے حب [ماڈلز](https://hf.co/docs/hub/models) دستاویزات دیکھیں۔

## ماڈل اپ لوڈ ہو رہا ہے

آپ کے ورک فلو کی ترجیح کے لحاظ سے حب پر ماڈل اپ لوڈ کرنے کے کئی طریقے ہیں۔ آپ ماڈل کو [`Trainer`] کے ساتھ دبا سکتے ہیں، [`~PreTrainedModel.push_to_hub`] کو براہ راست ماڈل پر کال کر سکتے ہیں، یا Hub ویب انٹرفیس استعمال کر سکتے ہیں۔

<Youtube id="Z1-XMy-GNLQ"/>

### ٹرینر

[`Trainer`] تربیت کے بعد ماڈل کو براہ راست حب کی طرف دھکیل سکتا ہے۔ `push_to_hub=True` کو [`TrainingArguments`] میں سیٹ کریں اور اسے [`Trainer`] پر منتقل کریں۔ تربیت مکمل ہونے کے بعد، ماڈل اپ لوڈ کرنے کے لیے [`~transformers.Trainer.push_to_hub`] کال کریں۔

[`~transformers.Trainer.push_to_hub`] خودکار طور پر مفید معلومات جیسے ٹریننگ ہائپر پیرامیٹر اور نتائج ماڈل کارڈ میں شامل کرتا ہے۔

```py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="my-awesome-model", push_to_hub=True)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.push_to_hub()
```

### پش ٹو ہب مکسین

[`~utils.PushToHubMixin`] ماڈل یا ٹوکنائزر کو حب کی طرف دھکیلنے کے لیے فعالیت فراہم کرتا ہے۔

اسے حب پر اپ لوڈ کرنے کے لیے [`~utils.PushToHubMixin.push_to_hub`] کو براہ راست ماڈل پر کال کریں۔ یہ [`~utils.PushToHubMixin.push_to_hub`] میں بیان کردہ ماڈل نام کے ساتھ آپ کے نام کی جگہ کے نیچے ایک ذخیرہ بناتا ہے۔

```py
model.push_to_hub("my-awesome-model")
```

دیگر اشیاء جیسے ٹوکنائزر کو بھی اسی طرح حب کی طرف دھکیل دیا جاتا ہے۔

```py
tokenizer.push_to_hub("my-awesome-model")
```

آپ کے Hugging Face پروفائل کو اب نئے بنائے گئے ماڈل ذخیرہ کو ظاہر کرنا چاہیے۔ اپ لوڈ کردہ تمام فائلز دیکھنے کے لیے **فائلز** ٹیب پر جائیں۔

فائلوں کو حب تک پہنچانے کے بارے میں مزید معلومات کے لیے [فائلوں کو حب میں اپ لوڈ کریں](https://hf.co/docs/hub/how-to-upstream) گائیڈ سے رجوع کریں۔

### حب ویب انٹرفیس

حب ویب انٹرفیس ماڈل اپ لوڈ کرنے کے لیے بغیر کوڈ کا طریقہ ہے۔

1. [**نیا ماڈل**](https://huggingface.co/new) کو منتخب کر کے ایک نیا ذخیرہ بنائیں۔

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/new_model_repo.png"/>
</div>

اپنے ماڈل کے بارے میں کچھ معلومات شامل کریں:

- ریپوزٹری کے **مالک** کو منتخب کریں۔ یہ آپ یا ان تنظیموں میں سے کوئی بھی ہو سکتا ہے جن سے آپ کا تعلق ہے۔
- اپنے ماڈل کے لیے ایک نام چنیں، جو کہ مخزن کا نام بھی ہوگا۔
- منتخب کریں کہ آیا آپ کا ماڈل عوامی ہے یا نجی۔
- لائسنس کا استعمال سیٹ کریں۔

2. ماڈل ذخیرہ بنانے کے لیے **Create ماڈل** پر کلک کریں۔

3. **فائلیں** ٹیب کو منتخب کریں اور فائل کو اپنے ذخیرہ میں گھسیٹنے اور چھوڑنے کے لیے **فائل شامل کریں** بٹن پر کلک کریں۔ ایک کمٹ میسج شامل کریں اور فائل کو کمٹ کرنے کے لیے **کمیٹ تبدیلیاں مین** پر کلک کریں۔

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/upload_file.png"/>
</div>

## ماڈل کارڈ

[ماڈل کارڈز](https://hf.co/docs/hub/model-cards#model-cards) صارفین کو ماڈلز کارکردگی، حدود، ممکنہ تعصبات، اور اخلاقی تحفظات کے بارے میں مطلع کرتے ہیں۔ اپنے ذخیرہ میں ماڈل کارڈ شامل کرنے کی انتہائی سفارش کی جاتی ہے!

ایک ماڈل کارڈ آپ کے ذخیرے میں ایک `README.md` فائل ہے۔ اس فائل کو بذریعہ شامل کریں:

- دستی طور پر ایک `README.md` فائل بنانا اور اپ لوڈ کرنا
- ریپوزٹری میں **ایڈیٹ ماڈل کارڈ** بٹن پر کلک کرنا

Llama 3.1 [ماڈل card](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) پر ایک نظر ڈالیں اس کی مثال کے لیے کہ ماڈل کارڈ میں کیا شامل کرنا ہے۔

[ماڈل کارڈز](https://hf.co/docs/hub/model-cards#model-cards) گائیڈ میں دستیاب دیگر ماڈل کارڈ میٹا ڈیٹا (کاربن کا اخراج، لائسنس، کاغذ سے لنک وغیرہ) کے بارے میں مزید جانیں۔
