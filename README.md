# Raspberry Bot
AI bot for multi-domain interactions
Team: Amit Jain, Cyril Chiffot

This project initially created as submission for the final project of Stanford Natural Lanfuage Understanding Course([XCS224U](https://online.stanford.edu/courses/xcs224u-natural-language-understanding))

The project aims to create a milti-domain task oriented dialog bot.
The project currently uses the `Schema-Guided Dialogue State Tracking (DSTC 8)` dataset defined in https://github.com/google-research-datasets/dstc8-schema-guided-dialogue.

The project tries to improve on the baseline in terms of performance, smaller model, less training time.

### Execution
```bash
pip install -r requirement.txt to install dependencies (Uses tensorflow gpu 1.15)
```

Following commands can be run for training, predictions using the associated split and then evaluate on the corresponding predictions.

#### Training
```bash
python -m raspberry.train_and_predict \
--model_ckpt_dir /data/models/albert/albert_base \
--dstc8_data_dir /data/dataset/dstc8-schema-guided-dialogue \
--output_base_dir /data/tasks/sgd/albert_base_vocaborg/single_domain \
--dataset_split train --run_mode train --model_initial_qualifier 0 \
--task_name dstc8_single_domain
```
#### Prediction
```bash
python -m raspberry.train_and_predict \
--dstc8_data_dir /data/dataset/dstc8-schema-guided-dialogue \
--output_base_dir /data/tasks/sgd/albert_base_vocaborg/single_domain \
--model_ckpt_dir /data/models/albert/albert_base \
--dataset_split dev --run_mode predict \
--task_name dstc8_single_domain \
--model_name albert-base-v2 \
--eval_ckpt 103235
```
#### Evaluation
```bash
python -m evaluate \
--dstc8_data_dir /data/dataset/dstc8-schema-guided-dialogue \
--prediction_dir /data/tasks/sgd/albert_base_vocaborg/single_domain/predictions/pred_res_103235_dev_dstc8_single_domain_dstc8-schema-guided-dialogue \
--output_metric_file /data/tasks/sgd/albert_base_vocaborg/single_domain/predictions/evaluations_103235.json \
--eval_set dev
```
### Initial commit - :
[7aedfb9ea913b391619a1634222111e52b775cdb](https://github.com/amit-jain/raspberry-bot/tree/7aedfb9ea913b391619a1634222111e52b775cdb)
* Changelog
    - Integrate ALBERT model in place of BERT
    - Some cosmetic changes to clean up
* Fine-tuning time
    - BERT - 11 hours
    - ALBERT - 9 hours
* Model Size
    - BERT - 1.4 GB
    - ALBERT - 260 MB
* Performance
    - *BERT*
    ```json
        "#ALL_SERVICES": {
            "active_intent_accuracy": 0.9678068410462777,
            "average_cat_accuracy": 0.6851462711166049,
            "average_goal_accuracy": 0.7760279721352729,
            "average_noncat_accuracy": 0.8117247518972562,
            "joint_cat_accuracy": 0.7061541304749585,
            "joint_goal_accuracy": 0.5061683936955064,
            "joint_noncat_accuracy": 0.6367003386988598,
            "requested_slots_f1": 0.9614871770304366,
            "requested_slots_precision": 0.9845545495193382,
            "requested_slots_recall": 0.9645372233400402
        },
        "#SEEN_SERVICES": {
            "active_intent_accuracy": 0.9895482130815914,
            "average_cat_accuracy": 0.8968550521563132,
            "average_goal_accuracy": 0.8863057910343625,
            "average_noncat_accuracy": 0.8939626305067481,
            "joint_cat_accuracy": 0.9034776437189496,
            "joint_goal_accuracy": 0.6983121712744438,
            "joint_noncat_accuracy": 0.7645259271746461,
            "requested_slots_f1": 0.9876151944257135,
            "requested_slots_precision": 0.9938750280962013,
            "requested_slots_recall": 0.9890424814565071
        },
        "#UNSEEN_SERVICES": {
            "active_intent_accuracy": 0.9462975316877918,
            "average_cat_accuracy": 0.44708508403361347,
            "average_goal_accuracy": 0.6645661263024276,
            "average_noncat_accuracy": 0.7263226232976332,
            "joint_cat_accuracy": 0.49170844581565754,
            "joint_goal_accuracy": 0.3160755170113409,
            "joint_noncat_accuracy": 0.5102391327551702,
            "requested_slots_f1": 0.9356380444105593,
            "requested_slots_precision": 0.9753335557038024,
            "requested_slots_recall": 0.9402935290193463
        }    
    ```
    * *ALBERT*
    ```json
        "#ALL_SERVICES": {
            "active_intent_accuracy": 0.8963782696177063,
            "average_cat_accuracy": 0.6938607334157396,
            "average_goal_accuracy": 0.7599752610674172,
            "average_noncat_accuracy": 0.785797658429007,
            "joint_cat_accuracy": 0.6873036407318426,
            "joint_goal_accuracy": 0.4752380865526492,
            "joint_noncat_accuracy": 0.5718519170020121,
            "requested_slots_f1": 0.946385455590687,
            "requested_slots_precision": 0.9920020120724347,
            "requested_slots_recall": 0.9471830985915493
        },
        "#SEEN_SERVICES": {
            "active_intent_accuracy": 0.9885367498314228,
            "average_cat_accuracy": 0.8840105869531371,
            "average_goal_accuracy": 0.8708555293912437,
            "average_noncat_accuracy": 0.8829710338680927,
            "joint_cat_accuracy": 0.8853797019162527,
            "joint_goal_accuracy": 0.670896493594066,
            "joint_noncat_accuracy": 0.7458086648685098,
            "requested_slots_f1": 0.9829175095527084,
            "requested_slots_precision": 0.99527983816588,
            "requested_slots_recall": 0.9833108563722185
        },
        "#UNSEEN_SERVICES": {
            "active_intent_accuracy": 0.8052034689793195,
            "average_cat_accuracy": 0.4800420168067227,
            "average_goal_accuracy": 0.6479044974524427,
            "average_noncat_accuracy": 0.6848853629512098,
            "joint_cat_accuracy": 0.47204010798303125,
            "joint_goal_accuracy": 0.281668094796531,
            "joint_noncat_accuracy": 0.3997519456304203,
            "requested_slots_f1": 0.9102433368277264,
            "requested_slots_precision": 0.9887591727818547,
            "requested_slots_recall": 0.9114409606404269
        }
    ```
 
As seen above, the model fine-tuned on ALBERT is underperforming slightly on some metrics, better on couple of metrics but major difference is in the performance of `active_intent_accuracy`. This underperformance is more significant in the unseen services as compared to the baseline performance using the BERT base cased model. But it is able to deliver this performance with a model which is smaller in size by a fifth. The model is uncompressed so, the actual compressed model maybe not show such a huge difference but will still be substantially smaller and would take 20% less time to train.

#### The key points affecting performance:
* Unavailability of a `cased` pre-trained ALBERT model. The nature of the task is such that the case of the tokens is an important signal for the model. Thus, this limitation accords severe disadvantage to the model. But this has not been confirmed by experiments due to lack of time. A simple yet sure way to confirm the theory would be to use an uncased BERT model to compare performance.
* Not related to performance but the initial experiments concentrated on integrating the HuggingFace transformers but using their pre-trained BERT/ALBERT models did not perform well at all due to a lurking bug in the implementations or an unknown configuration not not evident. This severly restricted the usage of other models from the repository available. Most promising of the models that were available is the DistillBert.

The various experiments performed:
* Using the BERT cased vocabulary and fine-tuning with case on. Gives quite a bit of performance loss hence, the approach was disbanded.
* Trying out different layers of the base ALBERT model for creating embeddings as suggested by authors of the BERT model. 
    - Last 4 layers - This approach has shown promise and helped improve the `active_intent_accuracy` to 93.8% but performance on various`goal_accuracy` and `noncat_accuracy` metrics dropped noticeably by 5% - 10%.
    - Hybrid (Last 4 layers for intent and last layer for others) - Does much worse (Hard to analyze but could be a problem in the implementation).
    - Second last layer

#### Future Work
* Larger model - The above approaches once are made to work either way can be tried for original baseline BERT model.
Another more promising avenue though is fine-tuning on the largers ALBERT pre-trained models e.g. ALBERT large, xlarge etc. Using these would still make for a substantially smaller model. Table 1 below shows comparison of the number of parameters for each.
In some aborted experiments it was found that the model for DSTC8 was ~ 450 MB using ALBERT large as compared to BERT base.
* Pre-trained ALBERT cased model - Getting hold off or pre-training a cased model should substantially improve performance as per current indications.
* Incorporating and experimenting with various pooling strategies for the first token embeddings e.g. taking mean, max for the other layers.
* Classification layer - Using RNN, LSTM etc. in the classification layers might also push performance.
* Data - The current experiments use only the single-domain data and training on the complete data presents more challenges but would better represent and generalize for real-world private datasets.
* Plugging the model to a dialog system like RASA, DeepPavlov etc.

### 2nd Commit 
[ae4cc3fe48477ece8ef0b9b1f6efce6b1da6efa6](https://github.com/amit-jain/raspberry-bot/tree/ae4cc3fe48477ece8ef0b9b1f6efce6b1da6efa6)
* Changelog
    - Last 4 layers for schema embeddings
* Performance with ALBERT
```json
    "#ALL_SERVICES": {
        "active_intent_accuracy": 0.9386317907444668,
        "average_cat_accuracy": 0.695879686856201,
        "average_goal_accuracy": 0.7235660932389896,
        "average_noncat_accuracy": 0.7392693455276642,
        "joint_cat_accuracy": 0.6854555535021253,
        "joint_goal_accuracy": 0.4359043450704225,
        "joint_noncat_accuracy": 0.5367620487357478,
        "requested_slots_f1": 0.9434727412091597,
        "requested_slots_precision": 0.9888006611095144,
        "requested_slots_recall": 0.9460093896713615
    },
    "#SEEN_SERVICES": {
        "active_intent_accuracy": 0.9865138233310856,
        "average_cat_accuracy": 0.8777829674606881,
        "average_goal_accuracy": 0.8502706000348856,
        "average_noncat_accuracy": 0.8619317545199899,
        "joint_cat_accuracy": 0.8697657913413769,
        "joint_goal_accuracy": 0.6358499662845584,
        "joint_noncat_accuracy": 0.7222795010114632,
        "requested_slots_f1": 0.9849966284558328,
        "requested_slots_precision": 0.9898853674983142,
        "requested_slots_recall": 0.987862440997977
    },
    "#UNSEEN_SERVICES": {
        "active_intent_accuracy": 0.8912608405603736,
        "average_cat_accuracy": 0.49133403361344535,
        "average_goal_accuracy": 0.5955011900354366,
        "average_noncat_accuracy": 0.6118872801798229,
        "joint_cat_accuracy": 0.4851523332047821,
        "joint_goal_accuracy": 0.23809289993328886,
        "joint_noncat_accuracy": 0.35322476939959974,
        "requested_slots_f1": 0.9023920709044125,
        "requested_slots_precision": 0.9877275326408083,
        "requested_slots_recall": 0.904603068712475
    }
```

Table 1

Model | Parameters | Layers | Hidden | Embedding | Parameter-sharing
--- | --- | --- | --- | --- | ---
BERT base | 108M | 12 | 768 | 768 | False
BERT large | 334M | 24 | 1024 | 1024 | False
ALBERT base | 12M | 12 | 768 | 128 | True
ALBERT large | 18M | 24 | 1024 | 128 | True
ALBERT xlarge | 60M | 24 | 2048 | 128 | True
ALBERT xxlarge | 235M | 12 | 4096 | 128 | True
