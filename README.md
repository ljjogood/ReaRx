## ReaRX

This repository is the implementation of ReaRx: Reasoning and Refinement for Safe Herbal Prescription Recommendation via Expert-Guided Reinforcement Learning.



## Data

- The TCM-GCD-example dataset comprises 100 patient samples with gastrointestinal malignancies.

- The institution‑specific guidelines include symptom–herb guidelines and disease–herb guidelines. We have provided a few sample guidelines in the data folder.

- Similar prescriptions are obtained via our predefined similarity‑based retrieval function, where for each query patient in TCM-GCD-example dataset, the prescription (including the herbal set and the corresponding dosages) of the most similar patient is retrieved.

  

Tip: To ensure data security, we have encoded the related data (i.e., each symptom/syndrome/disease/herb has been converted into an ID). The complete dataset from our study is available upon reasonable request from the corresponding author at [niujinghao2015@ia.ac.cn](mailto:niujinghao2015@ia.ac.cn).



## Quick start

run `python src/main.py`



## Contact

If you have any questions, please contact us via emaila: [ljjo9903@e.gzhu.edu.cn](mailto:ljjo9903@e.gzhu.edu.cn).

