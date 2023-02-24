# B-NER: A Novel Bangla Named Entity Recognition Dataset with Largest Entities and Its Baseline Evaluation

This is the implementation of our paper "B-NER: A Novel Bangla Named Entity Recognition Dataset with Largest Entities and Its Baseline Evaluation".

## Abstract
Within the Natural Language Processing (NLP) framework, Named Entity Recognition (NER) is regarded as the basis for extracting key information to understand texts in any language. As Bangla is a highly inflectional, morphologically rich, and resource-scarce language, building a balanced NER corpus with large and diverse entities is a demanding task. However, previously developed Bangla NER systems are limited to recognizing only three familiar entities: person, location, and organization. To address this significant limitation, we introduce a novel Bangla NER dataset B-NER, which was created using 22,144 manually annotated Bangla sentences collected from Bangla newspapers and Bangla Wikipedia. This dataset includes a total of 9,895 unique words which were manually categorized into eight different entity types, such as a person, organization, event, artifact, time indicator, natural phenomenon, geopolitical entity, and geographical location. Inter-annotator agreement experiments were conducted to validate the quality of annotations performed by three annotators, resulting in a Kappa score of 0.82. In this paper, we provide an outline of the annotation guideline illustrated with examples, discuss the B-NER dataset properties, and present benchmark evaluations of the dataset. In order to demonstrate the superiority and balance of the B-NER dataset compared to other publicly available datasets, we conducted a cross-dataset analysis. This analysis involved training the model on the B-NER dataset and testing it on publicly accessible datasets. The results showed that the model trained on B-NER performed optimally. Furthermore, we performed exhaustive benchmark evaluations based on Bidirectional LSTM with fastText embeddings and sentence transformer models. Among these models, fine-tuned IndicBERT achieved noticeable results with a macro accuracy of 86%. This dataset and baseline results will be publicly available under a CC-BY 4.0 license in the CoNLL-2002 format to facilitate further research on Bangla NER.

## Authors


* Md Zahidul Haque <sup>1</sup>
* Sakib Zaman <sup>1<sup>
* Jillur Rahman Saurav <sup>2<sup>
* Summit Haque <sup>3, 4<sup>
* Md Saiful Islam <sup>3, 5</sup>
* Mohammad Ruhul Amin <sup>6</sup>

<sup>1</sup> Sylhet Engineering College, Bangladesh
<br>
<br>
<sup>2</sup> University of Texas at Arlington, USA
<br>
<br>
<sup>3</sup> Shahjalal University of Science and Technology, Bangladesh
<br>
<br>
<sup>4</sup> Oregon State University, USA
<br>
<br>
<sup>5</sup> University of Alberta, Canada
<br>
<br>
<sup>6</sup> Fordham University, USA
<br>
<br>

## Installation
You're expected to have python 3.8 installed on your system.

Pull Data:
- `dvc pull`

With conda:
- `conda env create -f env.yaml` 

With pip:
- `pip install -r requirements.txt`

## Usage

Notebooks
- `ner_demo.ipynb --> for running the demo model`
- `train_ner.ipynb --> for training the model`
