# GilBERTo: Italian RoBERTa

**GilBERTo** is a pretrained language model for **Italian** based on [Facebook RoBERTa architecture](https://arxiv.org/abs/1907.11692).

Model was trained with the subword masking technique for 100k steps on ~71GB of **Italian text** with 11,250,012,896 words ([OSCAR](https://traces1.inria.fr/oscar/): **O***pen* **S***uper-large* **C***rawled* **A***LMAnaCH* *co***R***pus*). We use a vocabulary of 32k BPE subwords created using [SentencePiece](https://github.com/google/sentencepiece) tokenizer.

We evaluate GilBERTo in different downstream tasks, comparing with [mBERT](https://github.com/google-research/bert/blob/master/multilingual.md) and other (not BERT-based) models. Specifically we have compared the models in the following tasks:
* **P**art-**o**f-**S**peech tagging
* **N**amed **E**ntity **R**ecognition

## Download
**GilBERTo** is available both using *transfomers* and *fairseq* librarires.

Model | Library | Download
---|:---:|:---:
`GilBERTo-uncased` |*pytorch/fairseq* |[GilBERTo-uncased-fairseq.v1.zip](tbd)
`GilBERTo-uncased` |*huggingface/transformers* |[GilBERTo-uncased-transformers.v1.zip](tbd)

## Results
We're writing the paper with all details (*coming soon*). 

To the best of our knowledge, downstream task applications are limited due to the lack of datasets available for Italian.
**We strongly recommend everyone to contribute to the repository in order to improve the Italian NLP SOTA**. We will be happy to support.

Currently we selected the following tasks based on what we have found in the Italian state of the art: 

### PoS Tagging
PoS task has been evaluated using the Accuracy metric with two different Italian dataset: [Italian ParTUT](https://universaldependencies.org/treebanks/it_partut/index.html) and [Italian ISDT](https://universaldependencies.org/treebanks/it_isdt/index.html). We also compared the results with [**UDPipe** and **UDify**](https://arxiv.org/pdf/1904.02099.pdf) models.

Model | Italian ParTUT | Italian ISDT
:---:|:---:|:---:
UDPipe|98.4|98.4
UDify|98.2|98.5
mBERT|98.0|98.5
GilBERTo|**98.8**|**98.6**

### Named Entity Recognition
NER task has been evaluated using the [WikiNER Italian dataset](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500) already used by [Spacy pretrained model for Italian](https://spacy.io/models/it) who achieve `F-1 Score:86.40; Precision:86.73; Recall:86.08` 

Model | F1 | Precision | Recall
:---:|:---:|:---:|:---:
mBERT|92.2|92.1|92.3
GilBERTo|**92.7**|**92.7**|**92.8**


## How to use
You can simply use **GilBERTo** with the latest version of [huggingface/transformers](https://github.com/huggingface/transformers) or [pytorch/fairseq](https://github.com/pytorch/fairseq) Python libraries.
### huggingface/transformers
```python
from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("idb-ita/tbd", do_lower_case=True)
model = AutoModel.from_pretrained("idb-ita/tbd")

input_ids = torch.tensor(tokenizer.encode("Io sono italiano e mi chiamo GilBERTo!")).unsqueeze(0)  
#>> tensor([[5, 755, 181, 1413, 25, 155, 12513, 14397, 16247, 31976, 6]])
token_list = tokenizer.convert_ids_to_tokens(tokenizer.encode("Io sono italiano e mi chiamo GilBERTo!")) 
#>> ['<s>', '▁io', '▁sono', '▁italiano', '▁e', '▁mi', '▁chiamo', '▁gil', 'berto', '!', '</s>']

```
### pytorch/fairseq

    $ pip install fairseq
    
```python
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer

# Import GilBERTo with pytorch\fairseq Library
gilberto_model = FairseqRobertaModel.from_pretrained('path/to/checkpoints_folder', 
                                                    bpe='sentencepiece') 
# Mask Predictions
gilberto_model.fill_mask('Buongiorno mi <mask> Gilberto!', topk=3) #Fill mask token with GilBERTo

# Outputs
[('Buongiorno mi chiamo Gilberto!', 0.5044017434120178, ' chiamo'),
 ('Buongiorno mi presento Gilberto!', 0.05189879611134529, ' presento'),
 ('Buongiorno mi sento Gilberto!', 0.022937586531043053, ' sento')]
 
# Other examples

# Input: `È più facile per un italiano gesticolare senza <mask> che parlare senza gesticolare.`
# Output: `È più facile per un italiano gesticolare senza parlare che parlare senza gesticolare.`

# Input: `Agli italiani piace pasta, <mask> e mandolino`
# Output: `Agli italiani piace pasta, pizza e mandolino`

# Input: `Chi dice che il denaro non fa la <mask>, oltre a essere antipatico, è pure fesso.`
# Output: `Chi dice che il denaro non fa la felicità, oltre a essere antipatico, è pure fesso.`

# Input: `Era un uomo così antipatico che dopo la sua <mask> i parenti chiesero il bis`
# Output: `Era un uomo così antipatico che dopo la sua morte i parenti chiesero il bis`
```


## Contacts
**Giulio Ravasio**: [Linkedin](https://www.linkedin.com/in/giulio-ravasio-3a81a9110/) | [Twitter](https://twitter.com/GiulioRavasio) | [Github](https://github.com/giuliorav) | [e-mail](giulio.rav@gmail.com)

**Leonardo Di Perna**: [Linkedin](https://www.linkedin.com/in/leonardo-di-perna/) | [Twitter](https://twitter.com/Leodipe94) | [Github](https://github.com/LeoDeep) | [e-mail](dipernaleonardo@gmail.com)

## References
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* [Sentencepiece: A simple and language independent subword tokenizer and detokenizer for neural text processing](https://www.aclweb.org/anthology/D18-2012/)
* [Asynchronous Pipeline for Processing Huge Corpora on Medium to Low Resource Infrastructures](https://hal.inria.fr/hal-02148693)
* [CamemBERT: a Tasty French Language Model](https://www.researchgate.net/publication/337183733_CamemBERT_a_Tasty_French_Language_Model) (*it gave us inspiration*)
* [Learning multilingual named entity recognition from Wikipedia](https://figshare.com/articles/Learning_multilingual_named_entity_recognition_from_Wikipedia/5462500)
* [75 Languages, 1 Model: Parsing Universal Dependencies Universally](https://arxiv.org/abs/1904.02099)
* [Rajpurkar et al. 2016] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang SQuAD: 100,000+ Questions for Machine Comprehension of Text In the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP) — November 1–5, 2016 — Austin, Texas, USA.
* [Chen et al. 2017] Danqi Chen, Adam Fisch, Jason Weston and Antoine Bordes Reading Wikipedia to Answer Open-Domain Questions. In the Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), 2017 Vancouver
* *[Croce et al. 2018] Danilo Croce, Alexandra Zelenanska, Roberto Basili Neural Learning for Question Answering in Italian. AI*IA 2018 -- Advances in Artificial Intelligence, 2018 Trento. pages 389-402
