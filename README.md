# wiki_cheat application 

Link to the video - https://www.loom.com/share/b64de62586284a87adf23a83c697415d?sid=a8053b53-60fe-4a7f-8fe1-26e68177fcdd 

# Talk Track

Today we will go over the details of an application that is called WikiCheat. Users can ask questions to this application and app will answer the question, produce an evidence wiki passage that contains the answer and also produce a wikipedia link. The code is mostly organized in python notebooks and more emphasis is given to the overall modelling exercise.

Now lets directly jump to the app and run it across a bunch of queries. The overall architecture is divided into 3 major parts.

1) Retriever Module
2) Reranker Module
3) Reader Module

The retriever module takes in the user query and fetches the 100 most relevant results from our vector store. We are using a Bi-Encoder model that we trained on our dataset to encode the passages into 768 dimensional vector space. We will go over the experiment details for this Bi Encoder, later in the video. Now, once we have the top 100 results for our query, we send them to our reranker module and it reranks them. We are using a pretrained cross encoder to rerank our results. Once we have our re ranked results, we send the top 10 results to the Reader module which uses gpt4 turbo to further shortlist the passage that contains the answer to query. We take the challenge one step forward where we also produce the answer to query instead of just producing the passage that contains the answer.

As you might see, the model works in most of the cases barring some. This is still a proof of concept and requires a lot of enhancements and improvements to productionize it.

## Datasets

There are a bunch of public datasets that we can leverage here. We end up using a combination

1) wikiQA - Has close to 3000 question and passage pairs
2) wikipedia- 22-12-small- It has About 100k wikipedia articles divided into ~485k passages. We generate a question for 10,000 passages taken, each taken from a different article using Mistral 7B Instruct v0.2 model which was recently released by Mistral AI.

## Generate Training data

We are working on a A10 Nvidia GPU node which has 24GB of GPU memory. We were able to load the Mistral 7B model in bf16 precision but it was roughly taking 1 min to generate 5 question and answer pairs.

Then I stumbled upon OctoAI. They have a hosted a bunch of models on cloud and we can use them on per usage basis.

When I used their hosted Mistral 7B model, The throughput doubled. Now I was generating about 10 QnA pairs per min. This is still slow. It would have taken about 16 hours to just generate this dataset.

I played around with the prompt a bit and got a good throughput of ~80 QnA per min. Earlier we were asking it to generate 2 questions ( and easy and a difficult one ). The difficult question would have involved multi hop reasoning over the passage itself.
It took about 1 Dollar on their platform to generate this synthetic dataset.

We could have further increased the difficulty by sending in multiple passages from the same article and then asking it to generate a question which would involve multi hop reasoning across passages. And the same could be extended at an article level as well. For this exercise, we are generating a simple question

## Encode and Upload

I explored both Qdrant and Pinecone to store our embeddings. We could have used ann libs like faiss or scann, but creating multiple indexes on my local machine with 500k dps would have slowed down the experiment cycle. Hence I stuck to using the hosted versions of Qdrant and Pinecone. Qdrant in its free tier provides a small machine which has 1GB of RAM. I was not able to index the complete 500k passages on their free tier.

Pinecone serverless seemed to be a good choice for me because they were offerring a 100 dollar credit to new users and it just took about 10 dollars for all the read/writes and storage of embeddings on their platform.

It took about 20 mins to upload and create the index. I would have loved to explore and experiment with the vector store more. tinkering around with the indexing algorithm, tune the retrieval based on latency v/s recall, etc but due to limited time, that is out of scope for this exercise.

## Pipeline experiments

Now Lets look at some experiments that we did to produce our pipeline.

We are using 2000 questions as our test set from the 10k synthetic questions that we created earlier and we have indexed all the 480k wikipedia passages in our vector store.

a) V1.0 We first start by selecting a pre-trained sentence embedding model. There are a couple of public benchmarks that we can refer to. There is this MTEB benchmark and then the sentence transformers benchmark. We chose 'all_mpnet_base_v2' as our base sentence embedding model. There are other open source models such as bge and the e5 family. From my past experience, it was difficult to fine tune the bge models with a limited dataset. The e5 and the mpnet family of models were almost at par with each other when fine tuned. So for this exercise, I chose the all_mpnet_base_v2 model, but given more time and resources, I would have loved to see how the e5 family models perform on our dataset.

We were getting 0.7 as the top 1 recall.

b) v1.1 I then used a pretrained cross encoder model which was trianed a large passage retrieval dataset with user queries from bing search engine annoted with relevant text passages.

This increased our top 1 recall from 0.7 to 0.87 and

c) v2.0 I then decided to fine tune our sentence embedder model. Using the fine tuned sentence embedder model we were able to push the top 1 recall from 0.7 to 0.8 as compared to our base embedder model

d v2.1 and e v2.2 ). I fine tuned the cross encoder model as well. Surprisingly the trained model was performing sub par as compared to the original re ranker model on the entire corpus.

When we used the fine tuned re ranker to rank the results of our fine tuned retriever, we got a top 1 recall of 0.86, where as using the original re ranker gave a top 1 recall of 0.88.

This might be because of the small train dataset size, but this is something that I need to investigate further.

V2.3
We then used a LLM to further find the right passage from the top 10 passages from the reranking stage. We experimented with Mistral 7B and GPT4-turbo.

mistral 7 B
We faced multiple issues with mistral 7B. It kept adding extra pieces of information between answers, did not limit its answer to the best candidate answer, It would return multiple answers even when explicitly asked in the input prompt , It was also not able to understand if the retrieved passages did not have the right answer in them.

GPT4
We experimented with a couple of prompts and for the first 100 questions on our test set, we got 0.82 as the recall. The top 1 recall after reranking stage for the first 100 questions orginally was 0.83, so there is a minor drop in results.

This could be because of multiple reasons.

1) evaluation on a limited first 100 dataset,
2) The prompting techniques could be improved. We could use Chain of thought prompting with in context learning. We can also randomly shuffle the reranked options and do multiple runs across all of them and take a majority vote. This was recently introduced in the med prompt paper where authors showed that sometimes, LLMs are biased towards a particular option when they are presented multiple choice question.
3) A more in depth qualitative analysis would provide more details. From the initial looks of it, there are some instances where the passage chosen by the LLM is also correct. Now this could be attributed to the way we have generated our synthetic dataset. We are generating 1 question for 1 passage from an article. In this process we are not forcing the model to generate a question whose answer could only be found in the given passage but cannot be found in the other passages. This is also something that I would love to explore further.

## Train Embedder

Now lets look at some experiments that we did to fine tune our embedder model. Due to limited time, we did not perform an exhaustive search for the hyper parameters, loss function, learning rate, drop out etc. We are using the default learning rate of 2e-5 with a weight decay of 0.1.

We are using the MultipleNegativesRankingLoss as our loss function. Lets quickly understand how this loss function works.

This loss expects as input a batch consisting of sentence pairs (a_1, p_1), (a_2, p_2)…, (a_n, p_n) where we assume that (a_i, p_i) are a positive pair and (a_i, p_j) for i!=j a negative pair. For each a_i, it uses all other p_j as negative samples, i.e., for each a_i, we have 1 positive example (p_i) and n-1 negative examples (p_j). It then minimizes the negative log-likehood for softmax normalized scores.

We played with the batch size, experimented by adding a hard negative while preparing our train examples and also appended the title of the document with the passage text.

Here we have a test set of 2000 questions that we generated and have about 200 questions form the wikiqa dataset as part of our testset.

The model trained on just the synthetic dataset performed the best. It was giving us top 1 recall as 0.859 as compared to 0.796 as the top1 recall of our baseline model without fine tuning.

Increasing the batch size with multiple negative ranking loss usually increases the performance of the model and that is what we observed as well. The performance slowly increased from 0.839 to 0.859 when we increased the batch size from 16 to 44. Experiments with batch sizes larger than 44 were giving me OOM exceptions.

If we just append the title in the passage text and keep the other configurations constant, the model performance reduces from 0.859 to 0.834. This is in a setting when we only add the title at the train time. If we add title to our passages at the test time, the performance of the same model further reduces to 0.819.

Given a larger GPU and more time, we can perform much more elaborate experiments.

## Future Improvements

### Query Reformulation and Dataset generation
#### 1. We can rewrite the users queries and further decompose them to find answers that could be present 
####    across multiple passages and articles.
#### 2. We can generate a synthetic dataset that requires the model to perform multi hop reasoning
#### 3. We can generate a much larger dataset.
#### 4. There could be other interesting ways of chunking the wikipedia articles, instead of just using passages. 
####    We can keep an overlap between multiple passages to further enhance context. 

### RAG pipeline
#### 1. We can introduce an augment step where we can generate a set of questions that need to be answered first
####    in order to find the answer to the original question
#### 2. Instead of showing the complete paragraph as the evidence, we can show the exact part of the paragraph which 
####    contains the answer 
#### 3. In the reader stage, instead of sending in 10 passages from k different articles, we can send all passages 
####    of these k different articles to provide a better context to the LLM
#### 4. We could find the embeddings of the entire wikipedia article and store them in our vector store. We then 
####    use a 2-step process to first narrow down on the correct article and then narrow down on the right passage 
####    in that article.

### Modelling
#### 1. Experiments with other Embedding models, loss functions, exhaustive hyper-parameter search
#### 2. Benchmarking using other commercial embedders. Ex - OpenAI just released their new v3 embedding model
#### 3. We can parse the wikipedia articles, extract their headings and use the section hierarchy to enhance the 
####    context of the passages.
#### 4. Instead of using the encoder based models for generating sentence embeddings, we can use an LLM(a decoder only)
####    to generate embeddings. There is a e5-mistral-7b-instruct model which was trained in this manner and is sitting
####    top of the embedding benchmarks
#### 5. We can use instruction fine tune a smaller LLM like phi-2 to perform the reader task instead of using GPT-4 there. 

### This field is evolving at such a rapid pace that, we are bombarded with new ideas and content every other day. 
### So lets keep our heads down, learn from the community and keep growing. Thank you. 





























