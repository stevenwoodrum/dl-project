**- Explicitly name which input features you used (item content, subject normalized name, etc) and any manipulation you performed prior to input to FNN (normalization, embedding, down-sampling, oversampling, etc).**



My recent iteration of this removes mmdocrag from training since it does not have binary outcomes in resp, I also remove all other non 0/1 rows.



I used two input features: the normalized\_name (AI Model Name) and the item (task for AI Model). For the normalized\_name input I standardized (lowercased and stripped) and mapped to an integer ID (0 for missing/unseen) up to 132, the ID matches a 16 dim row in the embedding table that represents the model. For the task input prior to input I turned each task into a vector with SBERT, taking each unique item once (12,632 items) and running through all-MiniLM-L6-v2, each text became 384 numbers capturing its meaning. The vectors were saved to item\_emb.npy and their order to item\_emb\_ids.csv. Long tasks were truncated since MiniLM reads at most 256 tokens. I set the SBERT vectors as trainable=false (frozen) since I am doing leave-one-benchmark out so training and held-out items stay in the same embedding space. 



**- Briefly describe your FNN (input size, number of hidden layers and approximate width of each layer).**



My FNN takes in a single vector of 384 (items) + 16 (names) = 400 numbers. It has 1 hidden layer with 64 units using ReLU activation for fast compute and to minimize any chance of vanishing gradient problem. 



**- Name the loss metric(s) used (Brier, Cross-Entropy, etc).**



For training loss: I use MSE on the sigmoid output averaged over each 1,024 row minibatch. 

For evaluation: I used brier as the loss metric since it is what the competition uses. Brier being the average squared distance between predictions and outcomes. Since im doing one benchmark heldout I compute the brier for each benchmark being heldout and average it. 



**matharena            0.2670**

**multi\_swebench       0.3104**

**real\_webagents       0.2839**

**researchcodebench    0.3113**

**swe\_rebench          0.2751**

**Name: neural\_net, dtype: float32**



**avg (5 folds): 0.2895**



**- For each held-out benchmark, generate a plot of training and validation loss at each epoch.**



Attached (validation is the 10% of training benchmark items used for early stopping).



**- For the final trained model (no benchmarhjk held out), generate a plot of training and validation loss at each epoch.**



Attached (validation is the 10% of training benchmark items used for early stopping).



**- Explicitly name which optimization methods you used (e.g. Adam, early stopping, learning rate tuning). Optional: Compare validation loss of models with and without optimization.**



I used Adam (default and not tuned) and early stopping on validation MSE.



**- Send me a copy of the text file(s) from Codabench containing evaluation results. Will enable me to calculate the standard deviation of your Brier scores to assess stability.**



Attached



**- Your thoughts about the suitability of Approach 1 (FNN with input text embedding and zero-shot prediction) for the Stanford AIMS prediction task. Why it works, why it doesn't work, whether or not further tuning of Approach 1 would make a big difference vs Approach 1 being fundamentally incapable of high performance.**



I do not believe approach 1 has a suitable ceiling to our goals, so we should move on to approach 2. Approach 1 does achieve scoring items its never seen but it has a fundamental ceiling that I do not believe can be fixed with tuning. Approach 1 has no way to tell how hard a task is just what it is about, and this limits its potential. IRT in approach 2 could improve this fundamental issue by comparing ability against difficulty allowing for more accurate inference. Just due to the absence of data from the target benchmark at training, I am thinking TTA (approach 3) in the end will be most useful for accurate predictions. 



**- Optional: Send me serialized Keras History files from your training runs, which will enable me to generate custom plots.**



pushed to my git branch

