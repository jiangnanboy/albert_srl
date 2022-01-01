# albert-crf for SRL(Semantic Role Labeling)，中文语义角色标注

## 概述
自然语言的语义理解往往包括分析构成一个事件的行为、施事、受事等主要元素，以及其他附属元素（adjuncts），例如事件发生的时间、地点、方式等。
在事件语义学（Event semantics）中，构成一个事件的各个元素被称为语义角色（Semantic Roles）；而语义角色标注（Semantic Role Labeling）任务
就是识别出一个句子中所有的事件及其组成元素的过程，例如：其行为（往往是句子中的谓词部分），施事，事件，地点等。下图中，例子中的一个事件“生产”被识别出来，
而“生产”对应的施事“全行业”和受事“化肥二千七百二十万吨”以及附属的事件发生的时间“全年”被准确标注出来。语义角色标注可为许多下游任务提供支持，
例如：更深层的语义分析（AMR Parsing，CCG Parsing等），任务型对话系统中的意图识别，事实类问答系统中的实体打分等。

![image](https://raw.githubusercontent.com/jiangnanboy/albert_srl/master/image/example.png)

![image](https://raw.githubusercontent.com/jiangnanboy/albert_srl/master/image/label_table.png)

## 方法

利用huggingface/transformers中的albert+crf进行中文语义角色标注

利用albert加载中文预训练模型，后接一个前馈分类网络，最后接一层crf。利用albert预训练模型进行fine-tune。

整个流程是：

- 数据经albert后获取最后的隐层hidden_state=768
- 将last hidden_state=768和谓词位置指示器（predicate_indicator）进行concatenation，最后维度是（768 + 1）经一层前馈网络进行分类
- 将前馈网络的分类结果输入crf

![image](https://raw.githubusercontent.com/jiangnanboy/albert_srl/master/image/albert-crf-srl.png)

 ## 数据说明

BIOES形式标注，见data/

训练数据示例如下，其中各列为`字`、`是否语义谓词`、`角色`，每句仅有一个谓语动词为语义谓词，即每句中第二列取值为1的是谓词，其余都为0.

```
她 0 O
介 0 O
绍 0 O
说 0 O
， 0 O
全 0 B-ARG0
行 0 I-ARG0
业 0 E-ARG0
全 0 B-ARGM-TMP
年 0 E-ARGM-TMP
生 1 B-REL
产 1 E-REL
化 0 B-ARG1
肥 0 I-ARG1
二 0 I-ARG1
千 0 I-ARG1
七 0 I-ARG1
百 0 I-ARG1
二 0 I-ARG1
十 0 I-ARG1
万 0 I-ARG1
吨 0 E-ARG1
```

## 训练和预测见（examples/test_srl.py）

```
    srl = SRL(args)
    if train_bool(args.train):
        srl.train()
        '''
        epoch: 45, acc_loss: 46.13663248741068
        dev_score: 0.931195765914934
        val_loss: 50.842400789260864, best_val_loss: 50.84240078926086
        '''
    else:
        srl.load()
        # ner.test(args.test_path)
        text = '代表朝方对中国党政领导人和人民哀悼金日成主席逝世表示深切谢意'
        predicates_indicator = [0, 0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,1,1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0]
        assert len(text) == len(predicates_indicator)
        pprint(srl.predict(text, predicates_indicator))    
        # tag_predict: ['O', 'O', 'O', 'O', 'O', 'B-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'I-ARG0', 'E-ARG0', 'B-REL', 'E-REL', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'E-ARG1', 'O', 'O', 'O', 'O', 'O', 'O']
        # {'ARG0': '中国党政领导人和人民', 'ARG1': '金日成主席逝世', 'REL': '哀悼'}
```

## 项目结构
- data
    - example.dev
    - example.train
- examples
    - test_srl.py #训练及预测
- model
    - pretrained_model #存放预训练模型和相关配置文件
        - config.json
        - pytorch_model.bin
        - vocab.txt
- srl
    - utils
        - convert.py
    - dataset.py
    - model.py
    - module.py
- utils
    - log.py

## 参考
- [transformers](https://github.com/huggingface/transformers)
- [Simple BERT Models for Relation Extraction and Semantic Role Labeling](https://arxiv.org/pdf/1904.05255.pdf)