what I have learned (tried) for this competition: 

1. Deploy multiple tensorflow models into one graph (ensembling) - helps a lot (Final model is 0.879 from five 0.86 models)! The script can be found in ensemble_scripts -> merge_tensorflow_graph -> merge_models_into_one_graph.py

2. In part 1 above, we should choose to ensemble different model architectures - LSTM and VLAD models are quite diverse, thus gives better results when ensembling 

3. Tried using 2017 data - not helpful and only making results worse. Even after I only used samples whose labels are within the range for 2018's labels, it still only made the training worse. Conclusion - if two sets have different target distributions, don't join them. 

4. Have explored many, many different model variations for both LSTM and VLAD architectures - only slight improvements were discovered. When models' size are controlled within 1GB and below, the best model can only be found to be around 0.865. It's very hard to find a super-performing single model in this case. 

5. Augmentation - Have tried real time data augmentation by feeding only half (first or second half, each having 50% chance) of the entire video as one sample each time, as well as having 50% chance to reverse sample sequence order, during training phase. This, in theory, increases the data size to 4x of the original size, even though the augmented data samples are highly correlated within themselves. This only increases the performance for LSTM architecture not for VLAD type (of course!) from 0.855 to 0.863, and if, during inference, we feed in all six variations of input sequence (full seq, reversed full seq, first half seq, reversed first half seq, second half seq, and reversed second half seq) for each test sample, average the predicted probabilities, the result can be further boosted to 0.866!

6. I have found a videl-level-model that got single model performance of 0.842, from original 0.82x of Mixture-of-experts model. However, the increase of performance for video-level-model does not directly translate into increase of performance for frame-level-model, unfortunately. 

7. Based on the assumption that target labels are inter-correlated with each other, I tried using embeddings for each label, and doing a dot product for last dense layer with each label embeddings as the final scores before feeding into sigmoid. This improves, as mentioned above in 6, the video-level-model performance. I also tried to use average of the fasttext embeddings for the label key words as embeddings initializations. However, this makes no difference compared to randomly initialized the embeddings. 

8. Different ways of ensembling models - simply averaging the predicted probabilities for each model gives the best ensembling result.  

9. I noticed that there are many labels that are highly correlated (0.99 or above, or even at 1!) I manually changed those correlation to be exactly 1 during training, but found that it does not help improve the models' performance. We should not alter target labels distributions! 

10. Tried different loss functions, and found the default cross entropy loss to be best. Also tried negative sampling but didn't help. Also tried to add a rmse to loss for the predicted number of positive labels with actually number of positive labels, but that also didn't help. 

11. The only thing I didn't try is knowledge distillation, due to time and resource constraint. But that, unfortunately, were the approaches used by winning teams. 
