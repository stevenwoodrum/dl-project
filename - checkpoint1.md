**- Explicitly name which input features you used (item content, subject normalized name, etc) and any manipulation you performed prior to input to FNN (normalization, embedding, down-sampling, oversampling, etc).**



My recent iteration of this removes mmdocrag from training since it does not have binary outcomes in resp, I also remove all other non 0/1 rows.



I used two input features: the normalized\_name (AI Model Name) and the item (task for AI Model). For the task input prior to input I turned each task into a vector with SBERT, taking each unique item once (12,632 items) and running through all-MiniLM-L6-v2, each text became 384 numbers capturing its meaning. The vectors were saved to item\_emb.npy and their order to item\_emb\_ids.csv. Long tasks were truncated since MiniLM reads at most 256 tokens. I set the SBERT vectors as trainable=false (frozen) since I am doing leave-one-benchmark out and the held-out benchmark would not update. 



**- Briefly describe your FNN (input size, number of hidden layers and approximate width of each layer).**



My FNN takes in a single vector of 384 (items) + 16 (names) = 400 numbers. It has 64 hidden layers using ReLU activation for fast compute and to solve vanishing gradient problem. 



**- Name the loss metric(s) used (Brier, Cross-Entropy, etc).**



I used brier as the loss metric since it is what the competition uses. Brier being the average squared distance between predictions and outcomes. Since im doing one benchmark heldout I compute the brier for each benchmark being heldout and average it. 



**- For each held-out benchmark, generate a plot of training and validation loss at each epoch.**



Attached



**- For the final trained model (no benchmarhjk held out), generate a plot of training and validation loss at each epoch.**



Attached



**- Explicitly name which optimization methods you used (e.g. Adam, early stopping, learning rate tuning). Optional: Compare validation loss of models with and without optimization.**



I used Adam and early stopping.



**- Send me a copy of the text file(s) from Codabench containing evaluation results. Will enable me to calculate the standard deviation of your Brier scores to assess stability.**







**- Your thoughts about the suitability of Approach 1 (FNN with input text embedding and zero-shot prediction) for the Stanford AIMS prediction task. Why it works, why it doesn't work, whether or not further tuning of Approach 1 would make a big difference vs Approach 1 being fundamentally incapable of high performance.**



I do not believe approach 1 has a suitable ceiling to our goals, so we should move on to approach 2. Just due to the lack of useful training data I am thinking TTA (approach 3) in the end will be most useful for accurate predictions. 



**- Optional: Send me serialized Keras History files from your training runs, which will enable me to generate custom plots.**



pushed to my git branch

