This is the completed version of my FYP. 
This project retrieves the relevant articles in the PubMed database according to the seed corpus provided, and includes 3 different methods to achieve that: MC, MCMC and MLT.
nnPU is not included in this repository.
=========================================================================================================================================================================
Logs written in the winter vacation of 2024
## **Comparison with the original work of Monte-Carlo sampling:**
- [x] 1.__Seed corpus:__ I used the labeled positive sets in the `train.jsonl`
- [x] 2.__Keyword extraction:__ I extracted the keywords by tf-idf value rankings, also in `keywordExtraction.py`.
- [x] 3.__Query sampling:__ I used the Monte-Carlo sampling method with parameter sweeping in `keywordExtraction.py monte_carlo_with_parameter_sweeping()` method.
    - Here I only used the Monte Carlo sampling method in `keywordExtraction.py`. EXP method is implemented in file `KE_legacy`.
- [ ] 4.__Evaluation:__ The following work to be done is to retrieve the test set using the keywords extracted and measure the relevance to the domain using BM25 ranking function. I plan to use `ElasticSearch`.
  ![OpenAI Logo](C:\Users\YangG\Desktop\winterhomework.jpg)
- Several issues to be figured out
  
- [x] 1 What's the reason that Monte Carlo removes bias

According to the paper, the inclusion of a Monte-Carlo (MC) sampling procedure during the query string construction step in the proposed methodology offers two main benefits:

Decreased Human Expert Intervention: The use of the MC sampling procedure reduces the need for human expert intervention in the query process. Typically, expert involvement is an expensive and scarce resource. The approach can therefore help save on time and resources otherwise spent on human experts crafting and refining search queries.

Avoidance of Potential Human Bias: Human experts, while knowledgeable, can introduce bias into the search process based on their experiences, perspectives, or even unintentional preferences. The MC sampling method, by virtue of being algorithmic and probabilistic, can help avoid these biases. It constructs search strings by picking keywords from a ranked list with a probability distribution corresponding to their tf-idf weight, thus allowing for a more objective and balanced retrieval of documents.

The paper describes two case studies that demonstrate the effectiveness of this methodology in retrieving highly relevant papers without the need for expert domain knowledge, thereby supporting the use of MC sampling for unbiased and efficient information retrieval across various research fields​​.
- [x] 2 Why bother to use Monte-Carlo method to conduct sampling given that you can directly use the top 20 words with highest tfidf
 

Compared with the MC sampling, the chances are high that the results from this procedure are too general and do not correspond properly to the domain of interest.
(cited from section 6.2)

- [x] 3 Conduct another experiment using MLT with BM25 evaluation


## How to run the code:

1. Run `keywordExtraction.py`, which will generate `top_keywords.csv`, `terms_with_probabilities.csv`

2. Then run `databasequerying.py`, which queries the database testing.jsonl for performance evaluation.

## What else:
I am also running a separate file using MLT to conduct this task. After retrieval, an evaluation step containing the measurement of BM25 score will be employed and the results between these 2 methods will be compared.
