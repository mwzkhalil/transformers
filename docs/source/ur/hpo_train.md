<!--Copyright 2022 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# ہائپر پیرامیٹر کی تلاش

ہائپر پیرامیٹرز جیسے سیکھنے کی شرح، بیچ کا سائز، اور عہدوں کی تعداد تربیت کے نتائج کو نمایاں طور پر متاثر کرتی ہے۔ [`Trainer.hyperparameter_search`] متعدد ٹرائلز چلا کر، ہر ایک مختلف اقدار کے ساتھ، اور بہترین کو واپس کر کے بہترین امتزاج تلاش کرتا ہے۔

ہر ٹرائل `model_init` کے ساتھ ایک تازہ ماڈل شروع کرتا ہے، نئے ہائپر پیرامیٹر کے نمونے لیتا ہے، ایک مکمل ٹریننگ لوپ چلاتا ہے، اور سرچ بیک اینڈ کو ایک مقصد کی اطلاع دیتا ہے۔ بیک اینڈ ہر مقصد کو اگلی آزمائش سے آگاہ کرنے کے لیے استعمال کرتا ہے۔ تمام ٹرائلز مکمل ہونے کے بعد، بہترین ہائپر پیرامیٹر [`~trainer.utils.BestRun`] میں واپس کیے جاتے ہیں۔

## ماڈل شروع کیا جا رہا ہے

پچھلے رنز کی حالت سے بچنے کے لیے ہر آزمائش کو ایک تازہ ماڈل کے ساتھ شروع کریں۔ `model_init` ہر ٹرائل کے آغاز پر بلایا جاتا ہے اور ایک نیا ماڈل مثال دیتا ہے، اس لیے ہر آزمائش اسی ابتدائی وزن سے شروع ہوتی ہے۔

```py
from transformers import AutoModelForCausalLM

def model_init(trial):
    return AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
```

`model=` اور `model_init=` کو ایک ساتھ نہ پاس کریں یا [`Trainer`] ایک خرابی پیدا کریں۔

## تلاش کی جگہ کی وضاحت کریں۔

ایک فنکشن بنائیں جو تلاش کی جگہ کی وضاحت کرے۔ فارمیٹ پسدید پر منحصر ہے۔ اگر آپ `hp_space` فنکشن کی وضاحت نہیں کرتے ہیں تو، پہلے سے طے شدہ تلاش `learning_rate`، `num_train_epochs`، اور `per_device_train_batch_size` پر محیط ہے۔

```bash
# install one of these hyperparam search backends
pip install optuna
pip install wandb
pip install ray[tune]
```

<hfoptions id="backends">
<hfoption id="Optuna">

[Optuna](https://optuna.readthedocs.io/en/stable/index.html) ہائپر پیرامیٹر کی اصلاح کے لیے ایک ہلکا پھلکا فریم ورک ہے۔

```py
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [16, 32, 64, 128]),
    }
```

</hfoption>
<hfoption id="Ray Tune">

[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) ایک قابل توسیع ہائپر پیرامیٹر ٹیوننگ لائبریری ہے جو متعدد مشینوں میں ٹرائلز بھی تقسیم کر سکتی ہے۔

```py
from ray import tune

def hp_space(trial):
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }
```

</hfoption>
<hfoption id="Weights & Biases">

وزن اور تعصب](https://docs.wandb.ai/) بلٹ ان ہائپر پیرامیٹر تلاش کے ساتھ ایک تجرباتی ٹریکنگ پلیٹ فارم ہے۔ یہ Bayesian، بے ترتیب، اور گرڈ تلاش کی حکمت عملیوں کی حمایت کرتا ہے۔

```py
def hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "objective", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "per_device_train_batch_size": {"values": [16, 32, 64, 128]},
        },
    }
```

</hfoption>
</hfoptions>

## تلاش چلائیں۔

اصلاحی ہدف کی وضاحت کے لیے ایک اختیاری `compute_objective` فنکشن فراہم کریں۔ اگر موجود ہو تو یہ `eval_loss` پر طے شدہ ہے، یا بصورت دیگر تمام میٹرک اقدار کا مجموعہ۔ اس فال بیک پر انحصار کرنے سے بچنے کے لیے ایک واضح فنکشن پاس کریں۔ تلاش `backend` ایک دیے گئے `direction` میں `n_trials` رنز سے زیادہ مقصد کو بہتر بناتی ہے۔

```py
def compute_objective(metrics):
    return metrics["eval_loss"]

best_run = trainer.hyperparameter_search(
    hp_space=hp_space,
    compute_objective=compute_objective,
    n_trials=30,               # how many trials to run
    direction="minimize",      # or "maximize" for metrics like accuracy/F1
    backend="optuna",          # "optuna", "ray", or "wandb"
)
```

[`~Trainer.hyperparameter_search`] ایک [`~trainer.utils.BestRun`] لوٹاتا ہے جس میں معروضی قدر اور بہترین ہائپر پیرامیٹر امتزاج ہوتا ہے۔

```py
best_run = trainer.hyperparameter_search(...)

best_run.objective        # 0.38  (best eval loss)
best_run.hyperparameters  # {"learning_rate": 5e-5, "num_train_epochs": 4, ...}
```

[`TrainingArguments`] پر بہترین ہائپر پیرامیٹرز کا اطلاق کریں اور مکمل ڈیٹا سیٹ پر دوبارہ تربیت دیں۔
