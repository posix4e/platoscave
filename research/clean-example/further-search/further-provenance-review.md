# A stronger next language control: original ELMo

Reviewed 10 October 2026. This note does not change the frozen DINO/GloVe
experiment, its selected examples, or its protocol. No ELMo inference or
alignment scores were examined.

**Original ELMo is a good next candidate.** Its documented training corpus is
older and more narrowly specified than a modern web-trained sentence encoder,
and its word representations depend on their sentence context. This reduces
reliance on one bag-of-words representation. The comparison also changes the
corpus, objective and dimensionality, so it cannot isolate word order alone.
It is a stronger *control design*, not a claim that ELMo will score better.

AllenAI's [model catalog preserved in a November 2018 revision](https://github.com/allenai/allennlp-website/blob/40d800e5e6365520e8ae52f9d0302549fc11f1d5/elmo.html)
distinguishes **Original** from **Original (5.5B)**. The former has about
93.6 million parameters and was trained on the 1 Billion Word Benchmark.
The [benchmark publisher](https://www.statmt.org/lm-benchmark/) identifies its
source as WMT 2011 News Crawl. This independently connects the model provider's
corpus claim to the corpus provider's date, rather than inferring cleanliness
from ELMo's 2018 publication date. Our existing EPIC metadata audit places all
432 recordings in 2017. Those particular episodes and their later annotations
could not have appeared in that documented 2011 text collection.

The [original training repository](https://github.com/allenai/bilm-tf/tree/7cffee2b0986be51f5e2a747244836e1047657f4)
provides the language-model recipe and identifies the raw benchmark as its
training input. The selected recipe does not use an image-caption loss or an
external multimodal teacher. This does **not** establish that no ordinary
caption-like phrase occurs anywhere in the news corpus.

| Candidate | Verdict for another control |
| --- | --- |
| Original ELMo, 1B-Word model | Corpus chronology passes for EPIC-55 under published provenance. Highest priority for a second language encoder; original weight bytes still need downloading and SHA-256 pinning. |
| Original ELMo 5.5B | Keep separate. Its corpus adds Wikipedia; the inspected catalog does not specify that dump's date. Do not silently use the publisher's recommended default and inherit the original model's narrower provenance claim. |
| ELMo packaged inside an SNLI-trained task model | Exclude from a control intended to avoid explicit caption-derived training. [SNLI's own documentation](https://nlp.stanford.edu/projects/snli/) identifies Flickr30k captions and some Visual Genome material. This is not evidence of EPIC exposure; it is a different training-history objection. |
| Original DINO/GloVe/EPIC-55 | Still a valid first control for direct episode exposure. ELMo would add a language-model family and corpus change; it does not invalidate the existing control or justify replacing a disappointing result. |

The exact ELMo artifact is
`2x4096_512_2048cnn_2xhighway`, **without** `_5.5B`.
The [original options](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
were retrieved: 336 bytes, SHA-256
`86cb4c1a7e2f25e3867d06810e33ae27fbbd511ac77ca377be5fb7405a7085b1`.
They specify two bidirectional LSTM layers, 4096-dimensional cells and
512-dimensional directional projections, producing 1024-dimensional
contextual token outputs. The [original weights endpoint](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)
currently answers HTTP 200 and reports 374,434,792 bytes. **I did not download
or hash the full weights.** The server's 2017 Last-Modified header and multipart
ETag are recorded in `elmo-source-audit.json`; neither is an independent
historical attestation or a full artifact hash.

Two implementation traps matter before admitting an ELMo feature cache:

1. **Reset recurrent state between independent sentences/batches.** The
   [reference inference implementation](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/model.py)
   carries state across calls. Without explicit resets, a caption's vector can
   depend on preceding captions and their order. Test the same captions alone,
   in reordered batches, and in different batch sizes before scoring anything.
2. **Do not learn a layer mixture or borrow one from a task checkpoint.** The
   [layer-weighting helper](https://github.com/allenai/bilm-tf/blob/7cffee2b0986be51f5e2a747244836e1047657f4/bilm/elmo.py)
   creates trainable mixture/scaling parameters. A defensible first recipe is
   the frozen top contextual layer, mean-pooled over real tokens only, followed
   by L2 normalization. Preserve word order in the input; mask padding and
   exclude boundary markers. Freeze tokenizer, pooling and layer choice before
   any alignment outcome. Character input handles unseen tokens, but token
   length truncation and empty inputs still need reporting.

The remaining objection is substantial: this does not free the vision side
from human category selection. [ImageNet describes its organization around
WordNet synsets](https://www.image-net.org/about.php), and its
[collection paper](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf)
describes search and human verification. Ignoring classification labels in
DINO's loss does not undo that selection. ELMo's news also reflects human
language and interests. Shared object concepts, similar scenes, text-guided
image collection and research-time model selection remain possible sources of
correspondence.

Therefore the warranted claim is narrow: **documented independent unimodal
training on older corpora, tested on later recorded episodes**. It is not
“no shared human supervision,” “no common concepts,” or a proof of universal
representations. Keep the existing sample and its paired competence controls
when adding ELMo in a separately frozen extension; publish both encoders'
outcomes regardless of which succeeds.
