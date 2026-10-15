<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# CPU

CPU ٹریننگ اس وقت اچھی طرح کام کرتی ہے جب GPUs دستیاب نہ ہوں یا جب آپ لاگت سے موثر سیٹ اپ چاہتے ہوں۔ جدید Intel CPUs CPU بیک اینڈز کے لیے [PyTorch's AMP](https://docs.pytorch.org/docs/stable/amp) (خودکار مکسڈ پریسجن) کے ساتھ bf16 مخلوط صحت سے متعلق تربیت کی حمایت کرتے ہیں، جو میموری کے استعمال کو کم کرتا ہے اور تربیت کو تیز کرتا ہے۔

اگر ایک سی پی یو بہت سست ہے تو سی پی یو ٹریننگ کو متعدد ساکٹ یا نوڈس میں اسکیل کریں۔ ذیل کی مثالیں تین منظرناموں کا احاطہ کرتی ہیں:

- ایک سی پی یو
- ایک مشین پر متعدد عمل (ایک فی CPU ساکٹ)
- متعدد مشینوں میں متعدد عمل

تمام تقسیم شدہ مثالیں مواصلات کے لیے [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html) [Intel oneAPI HPC Toolkit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit.html) سے اور [`Trainer`] کے ساتھ DDP حکمت عملی کا استعمال کرتی ہیں۔

<hfoptions id="distrib-cpu">
<hfoption id="single CPU">

[`Trainer`] CPU پر bf16 مخلوط صحت سے متعلق تربیت کی حمایت کرتا ہے۔ CPU ٹریننگ کے لیے fp16 پر bf16 کو ترجیح دیں کیونکہ یہ عددی اعتبار سے زیادہ مستحکم ہے۔ PyTorch کے CPU آٹوکاسٹ کو فعال کرنے کے لیے `--bf16` اور CPU ٹریننگ کو مجبور کرنے کے لیے `--use_cpu` پاس کریں۔ آپ انہی پیرامیٹرز کو براہ راست [`TrainingArguments`] میں منتقل کر سکتے ہیں۔

```py
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./outputs",
    bf16=True,
    use_cpu=True,
)
```

</hfoption>
<hfoption id="distributed CPU (single node)">

ڈوئل ساکٹ سی پی یو پر، فی ساکٹ ایک عمل چلائیں۔ میموری کو ہر ساکٹ تک مقامی رکھنے سے تھرو پٹ بہتر ہوتا ہے۔ ذیل میں دی گئی مثال ایک مشین پر ایک عمل کے ساتھ ایک ساکٹ کے ساتھ دو عمل شروع کرتی ہے، جہاں `train.py` آپ کی [`Trainer`] ٹریننگ اسکرپٹ ہے۔

> [!TIP] > `OMP_NUM_THREADS` کو ایک ساکٹ پر فزیکل کور کی تعداد پر سیٹ کریں، مائنس ایک کور OS کے لیے مخصوص ہے۔ مثال کے طور پر، 24 کور ساکٹ سیٹ `OMP_NUM_THREADS=23` پر۔

```bash
export MASTER_ADDR=127.0.0.1
mpirun -n 2 -genv OMP_NUM_THREADS=23 \
python3 train.py \
 --output_dir ./outputs \
 --use_cpu \
 --bf16
```

</hfoption>
<hfoption id="distributed CPU (multiple nodes)">

ایک ہوسٹ فائل کا استعمال کرتے ہوئے دو Xeon مشینوں (`node0` اور `node1`) میں چار عملوں کی تربیت کا پیمانہ کریں جو ہر نوڈ کے IP ایڈریس کی فہرست دیتا ہے۔ `-n 4` جھنڈا عمل کی کل تعداد کا تعین کرتا ہے۔ `-ppn 2` فی نوڈ دو پراسیس سیٹ کرتا ہے (ایک فی ساکٹ)۔

اس اسکرپٹ کو `node0` سے چلائیں، جو کہ مرکزی عمل کے طور پر کام کرتا ہے۔

> [!TIP] > `OMP_NUM_THREADS` کو ایک ساکٹ پر فزیکل کور کی تعداد پر سیٹ کریں، مائنس ایک کور OS کے لیے مخصوص ہے۔ مثال کے طور پر، 24 کور ساکٹ سیٹ `OMP_NUM_THREADS=23` پر۔

ہر نوڈ کے IP ایڈریس کے ساتھ ایک ہوسٹ فائل بنائیں۔

```bash
cat hostfile
xxx.xxx.xxx.xxx #node0 ip
xxx.xxx.xxx.xxx #node1 ip
```

پھر ٹریننگ اسکرپٹ چلائیں۔

```bash
export MASTER_ADDR=xxx.xxx.xxx.xxx #node0 ip
mpirun -f hostfile -n 4 -ppn 2 \
 -genv OMP_NUM_THREADS=23 \
python3 train.py \
 --output_dir ./outputs \
 --use_cpu \
 --bf16
```

</hfoption>
</hfoptions>

## کوبرنیٹس

تقسیم شدہ CPU ٹریننگ [PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) کا استعمال کرتے ہوئے Kubernetes کلسٹر پر بھی چل سکتی ہے۔

تعینات کرنے سے پہلے درج ذیل سیٹ اپ کو مکمل کریں:

1. یقینی بنائیں کہ آپ کو [Kubeflow](https://www.kubeflow.org/docs/started/installing-kubeflow/) انسٹال کیے ہوئے Kubernetes کلسٹر تک رسائی حاصل ہے۔
2. کلسٹر کے ساتھ تعامل کرنے کے لیے [kubectl](https://kubernetes.io/docs/tasks/tools) انسٹال اور کنفیگر کریں۔
3. ڈیٹا سیٹس اور ماڈل فائلوں کو ذخیرہ کرنے کے لیے ایک [PersistentVolumeClaim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) سیٹ اپ کریں۔
4. ٹریننگ اسکرپٹ اور اس کے انحصار کے لیے ڈوکر امیج بنائیں۔

نیچے دی گئی مثال Dockerfile ایک Intel-optimized PyTorch بیس امیج سے شروع ہوتی ہے جس میں ملٹی نوڈ CPU ٹریننگ کے لیے MPI سپورٹ شامل ہے۔ یہ کارکردگی کی دو لائبریریاں بھی انسٹال کرتا ہے۔

- `google-perftools` (`libtcmalloc`): ایک میموری مختص کرنے والا جو پہلے سے طے شدہ سسٹم ایلوکیٹر کے مقابلے میں مختص اوور ہیڈ کو کم کرتا ہے۔
- `libomp-dev` (`libiomp5`): Intel کا OpenMP رن ٹائم، جو GNU OpenMP ڈیفالٹ سے بہتر CPU تھریڈ مینجمنٹ فراہم کرتا ہے۔

```dockerfile
FROM intel/intel-optimized-pytorch:2.4.0-pip-multinode

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    google-perftools \
    libomp-dev

WORKDIR /workspace

# Install Transformers and copy your training script into the image
RUN pip install --no-cache-dir transformers
COPY train.py /workspace/train.py
```

تصویر بنائیں اور اسے تعینات کرنے سے پہلے اپنے کلسٹر کے نوڈس سے قابل رسائی رجسٹری تک دھکیلیں۔

### پی ٹارچ جاب

[PyTorchJob](https://www.kubeflow.org/docs/components/training/user-guides/pytorch/) Kubernetes کا ایک حسب ضرورت وسیلہ ہے جو PyTorch تقسیم شدہ تربیتی ملازمتوں کا انتظام کرتا ہے۔ یہ کارکن پوڈ لائف سائیکل، دوبارہ شروع کرنے کی پالیسی، اور عمل کوآرڈینیشن کو سنبھالتا ہے۔ آپ کی ٹریننگ اسکرپٹ کو صرف ماڈل اور ڈیٹا کو ہینڈل کرنے کی ضرورت ہے۔

ذیل میں دی گئی مثال yaml فائل چار کارکنوں کو ترتیب دیتی ہے جو bf16 فعال کے ساتھ ایک ٹریننگ اسکرپٹ چلا رہے ہیں۔

CPU وسائل کی حدود اور درخواستیں متعین کرتے وقت، [CPU units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-cpu) استعمال کریں جہاں ایک یونٹ ایک فزیکل کور یا ورچوئل کور کے برابر ہو۔ `Guaranteed` [خدمت کا معیار](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod) کے لیے حد اور درخواست دونوں کو ایک ہی قیمت پر سیٹ کریں۔ کیوبلیٹ اور سسٹم کے عمل کے لیے کچھ کور غیر مختص چھوڑ دیں۔ `OMP_NUM_THREADS` کو مختص کردہ CPU یونٹس کی تعداد سے ملنے کے لیے سیٹ کریں تاکہ PyTorch تمام دستیاب کور استعمال کرے۔

یامل فائل کو اپنی ٹریننگ اسکرپٹ اور اپنے کلسٹر میں نوڈس کی تعداد کی بنیاد پر ڈھال لیں۔

```yaml
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: transformers-pytorchjob
spec:
  elasticPolicy:
    rdzvBackend: c10d
    minReplicas: 1
    maxReplicas: 4
    maxRestarts: 10
  pytorchReplicaSpecs:
    Worker:
      replicas: 4  # The number of worker pods
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: <image name>:<tag>  # Specify the docker image to use for the worker pods
              imagePullPolicy: IfNotPresent
              command: ["/bin/bash", "-c"]
              args:
                - >-
                  cd /workspace;
                  torchrun train.py \
                    --model_name_or_path distilbert/distilbert-base-uncased \
                    --do_train \
                    --do_eval \
                    --per_device_train_batch_size 12 \
                    --learning_rate 3e-5 \
                    --num_train_epochs 2 \
                    --output_dir /tmp/pvc-mount/output_$(date +%Y%m%d_%H%M%S) \
                    --bf16;
              env:
              - name: LD_PRELOAD
                value: "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4.5.9:/usr/local/lib/libiomp5.so"
              - name: HF_HUB_CACHE
                value: "/tmp/pvc-mount/hub_cache"
              - name: HF_DATASETS_CACHE
                value: "/tmp/pvc-mount/hf_datasets_cache"
              - name: LOGLEVEL
                value: "INFO"
              - name: OMP_NUM_THREADS  # Set to match the number of allocated CPU units
                value: "240"
              resources:
                limits:
                  cpu: 240  # Update the CPU and memory limit values based on your nodes
                  memory: 128Gi
                requests:
                  cpu: 240  # Update the CPU and memory request values based on your nodes
                  memory: 128Gi
              volumeMounts:
              - name: pvc-volume
                mountPath: /tmp/pvc-mount
              - mountPath: /dev/shm
                name: dshm
          restartPolicy: Never
          nodeSelector:  # Optionally use nodeSelector to match a certain node label for the worker pods
            node-type: gnr
          volumes:
          - name: pvc-volume
            persistentVolumeClaim:
              claimName: transformers-pvc
          - name: dshm
            emptyDir:
              medium: Memory
```

### تعینات کریں۔

PyTorchJob کو درج ذیل کمانڈ کے ساتھ کلسٹر میں تعینات کریں۔

```bash
export NAMESPACE=<specify your namespace>

kubectl create -f pytorchjob.yaml -n ${NAMESPACE}
```

پوڈز کو ان کی حیثیت کی نگرانی کے لیے نام کی جگہ پر درج کریں۔ پوڈز **پینڈنگ** کے طور پر شروع ہوتے ہیں جب کہ کنٹینر کی تصویر کھینچی جاتی ہے، پھر **رننگ** میں منتقلی ہوتی ہے۔

```bash
kubectl get pods -n ${NAMESPACE}

NAME                                                     READY   STATUS                  RESTARTS          AGE
...
transformers-pytorchjob-worker-0                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-1                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-2                         1/1     Running                 0                 7m37s
transformers-pytorchjob-worker-3                         1/1     Running                 0                 7m37s
...
```

تربیتی پیشرفت کی پیروی کرنے کے لیے ورکر پوڈ سے لاگز کو سٹریم کریں۔

```bash
kubectl logs transformers-pytorchjob-worker-0 -n ${NAMESPACE} -f
```

تربیت یافتہ ماڈل کو پی وی سی سے کاپی کریں یا ٹریننگ مکمل ہونے کے بعد اپنے اسٹوریج کی جگہ۔ پھر PyTorchJob وسائل کو حذف کریں۔

```bash
kubectl delete -f pytorchjob.yaml -n ${NAMESPACE}
```

## اگلے اقدامات

- جدید انٹیل ہارڈ ویئر پر BF16 کی کارکردگی پر گہری نظر کے لیے [Intel Sapphire Rapids](https://huggingface.co/blog/intel-sapphire-rapids) کے ساتھ Accelerating PyTorch Transformers بلاگ پوسٹ پڑھیں۔
