<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# انفرنس کے لیے torch.compile

[torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) PyTorch کوڈ کو آپٹمائزڈ کرنل میں مرتب کرتا ہے جو انفرنس کو نمایاں طور پر تیز کرتا ہے۔ یہ فیچر [TorchDynamo](https://pytorch.org/docs/stable/torch.compiler_dynamo_overview.html) کوڈ کو گراف میں مرتب کرنے کے لیے اور [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) پر انحصار کرتا ہے تاکہ گراف کو مزید بہتر کرنل میں مرتب کیا جا سکے۔ یہ ایک طاقتور اصلاحی ٹول ہے، اور بہت سے معاملات میں، کوڈ کی صرف ایک لائن کو شامل کرنے کی ضرورت ہوتی ہے۔

ایک ماڈل کو torch.compile کے ساتھ لپیٹیں اور ایک آپٹمائزڈ ماڈل کو واپس کریں۔

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model)
```

> [!TIP] > torch.compile پر ابتدائی کال سست ہے کیونکہ ماڈل کو مرتب کرنے کی ضرورت ہے۔ مرتب شدہ ماڈل پر آنے والی کالیں بہت تیز ہیں کیونکہ اسے دوبارہ مرتب کرنے کی ضرورت نہیں ہے۔

تالیف کے عمل کو حسب ضرورت بنانے کے لیے کئی پیرامیٹرز ہیں۔ ان میں سے دو زیادہ اہم ذیل میں درج ہیں۔ پیرامیٹرز کی مکمل فہرست کے لیے torch.compile [documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) سے رجوع کریں۔

## موڈز

`mode` پیرامیٹر مرتب کرنے کے لیے کارکردگی کے کئی اختیارات پیش کرتا ہے۔ یہ دیکھنے کے لیے مختلف طریقوں کو آزمائیں کہ آپ کے استعمال کے معاملے میں کون سا بہترین کام کرتا ہے۔

- `default` رفتار اور میموری کے درمیان ایک متوازن آپشن ہے۔
- `reduce-overhead` تھوڑی زیادہ میموری کی قیمت پر Python کے اوور ہیڈ کو کم کرتا ہے، لیکن یہ تیز تر ہو سکتا ہے۔
- `max-autotune` تیز ترین رفتار پیش کرتا ہے، لیکن تالیف میں زیادہ وقت لگتا ہے۔

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model, mode="reduce-overhead")
```

## فل گراف

فل گراف کارکردگی کو زیادہ سے زیادہ کرنے کے لیے پورے ماڈل کو ایک ہی گراف میں مرتب کرنے کی کوشش کرتا ہے۔ torch.compile اگر گراف بریک کا سامنا کرتا ہے تو ایک غلطی پیدا کرتا ہے، جس کا مطلب ہے کہ یہ ماڈل کو ایک گراف میں مرتب نہیں کر سکتا۔

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-2b", device_map="auto")
compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
```

## بینچ مارکس

مختلف وژن کے کاموں کے لیے ایک ہی تصویر پر مختلف GPUs اور بیچ سائز میں torch.compile کو فعال اور غیر فعال کے ساتھ ملی سیکنڈز میں اوسط انفرنس وقت کا موازنہ کرنے کے لیے کارکردگی کے معیارات کے لیے نیچے دیے گئے جدول کو دیکھیں۔

مختلف GPUs کے درمیان سوئچ کرنے کے لیے نیچے دیے گئے جدول میں **سب سیٹ** کو منتخب کریں، نیز بینچ مارکس کو [PyTorch nightly](https://download.pytorch.org/whl/nightly/cu118) 2.1.0dev اور torch.compile `reduce-overhead` موڈ کے ساتھ فعال کریں۔

<iframe
  src="https://huggingface.co/datasets/docs-benchmarks/compile-benchmarks/embed/viewer/A100/train"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>

## اگلے اقدامات

- تربیت کے لیے torch.compile کو استعمال کرنے کا طریقہ سیکھنے کے لیے [torch.compile for training](https://huggingface.co/docs/transformers/main/en/torch_compile) گائیڈ دیکھیں۔
