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
# کمپیوٹر وژن کے لیے علم کشید

[[open-in-colab]]

علم کشید ایک تکنیک ہے جو علم کو ایک بڑے، زیادہ پیچیدہ ماڈل (استاد) سے چھوٹے، آسان ماڈل (طالب علم) کو منتقل کرنے کے لیے استعمال کیا جاتا ہے۔ علم کو ایک ماڈل سے دوسرے تک پھیلانے کے لیے، ہم ایک پہلے سے تربیت یافتہ استاد ماڈل کو کسی خاص کام پر تربیت دیتے ہیں (اس معاملے کے لیے تصویر کی درجہ بندی) اور تصویر کی درجہ بندی پر تربیت یافتہ طالب علم ماڈل کو تصادفی طور پر شروع کرتے ہیں۔ اس کے بعد، ہم طالب علم کو تربیت دیتے ہیں ماڈل اس کے آؤٹ پٹس اور ٹیچر کے آؤٹ پٹس کے درمیان فرق کو کم کرنے کے لیے، اس طرح وہ رویے کی نقل کرتا ہے۔ یہ سب سے پہلے [ہنٹن ایٹ ال](https://huggingface.co/papers/1503.02531) کے ذریعہ نیورل نیٹ ورک میں علم کو کشید کرنا] میں متعارف کرایا گیا تھا۔ اس گائیڈ میں، ہم کام کے لیے مخصوص علم کشید کریں گے۔ ہم اس کے لیے [beans dataset](https://huggingface.co/datasets/AI-Lab-Makerere/beans) استعمال کریں گے۔

یہ گائیڈ ظاہر کرتا ہے کہ آپ کس طرح ایک [فائن ٹیونڈ ViT کو ڈسٹل کر سکتے ہیں۔ماڈل](https://huggingface.co/merve/vit-mobilenet-beans-224)(استادماڈلایک [موبائل نیٹ پر](https://huggingface.co/google/mobilenet_v2_1.4_224)(طالب علمماڈل) [Trainer API کا استعمال کرتے ہوئے](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#trainer)کا 🤗Transformers.

آئیے ڈسٹلیشن اور عمل کا جائزہ لینے کے لیے درکار لائبریریوں کو انسٹال کرتے ہیں۔

```bash
pip install transformers datasets accelerate tensorboard evaluate trackio --upgrade
```

اس مثال میں، ہم `merve/beans-vit-224` ماڈل بطور استاد ماڈل استعمال کر رہے ہیں۔ یہ ایک تصویر کی درجہ بندی ماڈل ہے، جس کی بنیاد `google/vit-base-patch16-224-in21k` پھلیوں کے ڈیٹاسیٹ پر فائن ٹیونڈ ہے۔ ہم اس ماڈل کو تصادفی طور پر شروع کردہ MobileNetV2 میں ڈسٹل کریں گے۔

اب ہم ڈیٹاسیٹ لوڈ کریں گے۔

```python
from datasets import load_dataset

dataset = load_dataset("AI-Lab-Makerere/beans")
```

ہم ماڈلز میں سے کسی ایک سے تصویری پروسیسر استعمال کر سکتے ہیں، جیسا کہ اس صورت میں وہ ایک ہی ریزولوشن کے ساتھ ایک ہی آؤٹ پٹ واپس کرتے ہیں۔ ڈیٹا سیٹ کے ہر اسپلٹ پر پری پروسیسنگ لاگو کرنے کے لیے ہم `dataset` کا `map()` طریقہ استعمال کریں گے۔

```python
from transformers import AutoImageProcessor
teacher_processor = AutoImageProcessor.from_pretrained("merve/beans-vit-224")

def process(examples):
    processed_inputs = teacher_processor(examples["image"])
    return processed_inputs

processed_datasets = dataset.map(process, batched=True)
```

بنیادی طور پر، ہم چاہتے ہیں کہ طالب علم ماڈل (ایک تصادفی طور پر شروع کیا گیا موبائل نیٹ) استاد ماڈل (فائن ٹیونڈ وژن ٹرانسفارمر) کی نقل کرے۔ اس کو حاصل کرنے کے لیے، ہم سب سے پہلے استاد اور طالب علم سے لاگٹس آؤٹ پٹ حاصل کرتے ہیں۔ پھر، ہم ان میں سے ہر ایک کو پیرامیٹر `temperature` سے تقسیم کرتے ہیں جو ہر نرم ہدف کی اہمیت کو کنٹرول کرتا ہے۔ `lambda` نامی پیرامیٹر ڈسٹلیشن نقصان کی اہمیت کا وزن کرتا ہے۔ اس مثال میں، ہم `temperature=5` اور `lambda=0.5` استعمال کریں گے۔ ہم طالب علم اور استاد کے درمیان فرق کی گنتی کے لیے Kullback-Leibler Divergence نقصان کا استعمال کریں گے۔ دو ڈیٹا P اور Q کو دیکھتے ہوئے، KL Divergence وضاحت کرتا ہے کہ Q کا استعمال کرتے ہوئے P کی نمائندگی کرنے کے لیے ہمیں کتنی اضافی معلومات کی ضرورت ہے۔ اگر دو ایک جیسے ہیں، تو ان کا KL ڈائیورجنس صفر ہے، کیونکہ P سے Q کی وضاحت کے لیے کسی اور معلومات کی ضرورت نہیں ہے۔ اس طرح، علم کشید کے تناظر میں، KL ڈائیورجنس مفید ہے۔

```python
from transformers import TrainingArguments, Trainer
from accelerate import Accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageDistilTrainer(Trainer):
    def __init__(self, teacher_model=None, student_model=None, temperature=None, lambda_param=None,  *args, **kwargs):
        super().__init__(model=student_model, *args, **kwargs)
        self.teacher = teacher_model
        self.student = student_model
        self.loss_function = nn.KLDivLoss(reduction="batchmean")
        device = Accelerator().device
        self.teacher.to(device)
        self.teacher.eval()
        self.temperature = temperature
        self.lambda_param = lambda_param

    def compute_loss(self, student, inputs, return_outputs=False):
        student_output = self.student(**inputs)

        with torch.no_grad():
          teacher_output = self.teacher(**inputs)

        # Compute soft targets for teacher and student
        soft_teacher = F.softmax(teacher_output.logits / self.temperature, dim=-1)
        soft_student = F.log_softmax(student_output.logits / self.temperature, dim=-1)

        # Compute the loss
        distillation_loss = self.loss_function(soft_student, soft_teacher) * (self.temperature ** 2)

        # Compute the true label loss
        student_target_loss = student_output.loss

        # Calculate final loss
        loss = (1. - self.lambda_param) * student_target_loss + self.lambda_param * distillation_loss
        return (loss, student_output) if return_outputs else loss
```

اب ہم Hugging Face Hub میں لاگ ان کریں گے تاکہ ہم اپنے ماڈل کو `Trainer` کے ذریعے Hugging Face حب کی طرف دھکیل سکیں۔

```python
from huggingface_hub import notebook_login

notebook_login()
```

آئیے `TrainingArguments`، استاد ماڈل اور طالب علم ماڈل سیٹ کریں۔

```python
from transformers import AutoModelForImageClassification, MobileNetV2Config, MobileNetV2ForImageClassification

training_args = TrainingArguments(
    output_dir="my-awesome-model",
    num_train_epochs=30,
    fp16=True,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    hub_strategy="every_save",
    hub_model_id=repo_name,
    report_to="trackio",
    run_name="distillation",
    )

num_labels = len(processed_datasets["train"].features["labels"].names)

# initialize models
teacher_model = AutoModelForImageClassification.from_pretrained(
    "merve/beans-vit-224",
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# training MobileNetV2 from scratch
student_config = MobileNetV2Config()
student_config.num_labels = num_labels
student_model = MobileNetV2ForImageClassification(student_config)
```

ہم ٹیسٹ سیٹ پر اپنے ماڈل کا اندازہ کرنے کے لیے `compute_metrics` فنکشن استعمال کر سکتے ہیں۔ یہ فنکشن ہمارے ماڈل کے `accuracy` اور `f1` کی گنتی کے لیے تربیتی عمل کے دوران استعمال کیا جائے گا۔

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    acc = accuracy.compute(references=labels, predictions=np.argmax(predictions, axis=1))
    return {"accuracy": acc["accuracy"]}
```

آئیے `Trainer` کو ان تربیتی دلائل کے ساتھ شروع کریں جن کی ہم نے وضاحت کی ہے۔ ہم اپنے ڈیٹا کولیٹر کو بھی شروع کریں گے۔

```python
from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()
trainer = ImageDistilTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    training_args=training_args,
    train_dataset=processed_datasets["train"],
    eval_dataset=processed_datasets["validation"],
    data_collator=data_collator,
    processing_class=teacher_processor,
    compute_metrics=compute_metrics,
    temperature=5,
    lambda_param=0.5
)
```

اب ہم اپنی ماڈل کو تربیت دے سکتے ہیں۔

```python
trainer.train()
```

ہم ٹیسٹ سیٹ پر ماڈل کا اندازہ لگا سکتے ہیں۔

```python
trainer.evaluate(processed_datasets["test"])
```

ٹیسٹ سیٹ پر، ہمارا ماڈل 72 فیصد درستگی تک پہنچ جاتا ہے۔ ڈسٹلیشن کی کارکردگی پر سنجیدگی سے جانچ کرنے کے لیے، ہم نے اسی ہائپر پیرامیٹر کے ساتھ شروع سے ہی سیم ڈیٹا سیٹ پر موبائل نیٹ کو تربیت دی اور ٹیسٹ سیٹ پر 63 فیصد درستگی کا مشاہدہ کیا۔ ہم قارئین کو دعوت دیتے ہیں کہ وہ مختلف پہلے سے تربیت یافتہ استاد ماڈلز، طلباء کے فن تعمیر، ڈسٹلیشن پیرامیٹرز کو آزمائیں اور اپنے نتائج کی اطلاع دیں۔ ڈسٹلڈ ماڈل کے لیے ٹریننگ لاگز اور چیک پوائنٹس [اس ریپوزٹری](https://huggingface.co/merve/vit-mobilenet-beans-224) میں مل سکتے ہیں، اور شروع سے تربیت یافتہ MobileNetV2 اس [repository](https://huggingface.co/merve/resnet-mobilenet-beans-5) میں مل سکتے ہیں۔
