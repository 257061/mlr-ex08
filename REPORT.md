Task 4 – Improving the Leaf Classifier

1. Introduction
-The original model from Task 2 works on a phone, but it does not always recognise leaves correctly. The training images are small and already cropped, while real phone camera images can have different lighting, angles and backgrounds.
-The goal was to find out which changes improve the model and which do not.
-I tested different training methods and compared their results with the original model. Each experiment was repeated three times with different random seeds to make the results more reliable.

2. Experiments and results
-The dataset contains 244 training images and 62 validation images of four leaf types: apple, cherry, chestnut and maple.
-I tested the models on three datasets:
Validation: the original 62 images.
Camera-like: the same images with different crops, rotations, colours and blur.
Last year: 120 images from other students, used to test how well the model works with new photos.
-The table shows the average accuracy from three runs. The number in brackets shows the difference between the best and worst run.

Experiment	Change	Validation	Camera-like	Last year
Baseline	Original model	0.769	0.694	0.517
ColorJitter	Changed brightness, contrast and colours	0.801	0.737	0.586
Geometric	Added crops, flips and rotations	0.817	0.677	0.583
Geometric + Color	Combined both changes	0.839	0.769	0.628
Frozen backbone	Trained only the final layer	0.790	0.629	0.464

3. What did the changes improve?
-Color changes (ColorJitter) improved all three results. The camera-like accuracy increased from 0.694 to 0.737. Results on last year's photos also improved from 0.517 to 0.586. This suggests that the model benefits from learning to recognise leaves under different lighting conditions.
-Geometric changes improved the normal validation result, but did not improve the camera-like result. This means that cropping and rotation alone were not very helpful.
-Combining both changes gave the best results so far. Accuracy increased to 0.839 on validation images, 0.769 on camera-like images and 0.628 on last year's images.
-Freezing the backbone made training much faster (14 seconds instead of 47 seconds), but the model became less accurate. It performed worse than the original model on both camera-like and last-year images.

4. Is the improvement bigger than random variation?
-The results are not always stable. For example, the original model had a validation accuracy between 0.710 and 0.823 in its three runs.
-This means that some small improvements may simply be caused by random differences between runs.
-The combined augmentation gave the best overall results, but the differences between runs are still quite large. Therefore, more testing would be useful before claiming that every improvement is significant.
-The biggest and clearest negative result is freezing the backbone: it was faster, but its accuracy was worse.

5. What did not help?
Training only the final layer was faster but reduced accuracy.
Geometric augmentation alone did not improve the camera-like results.
Using only validation accuracy is not enough, because performance on new photos can be much worse.
The last training epoch was often worse than the best epoch, so keeping the best model may be useful.

6. Final recommendation
-Based on the results, I would choose the model trained with both colour and geometric augmentation.
-It achieved the best results on all three test sets and seems more suitable for different photo conditions.
-I would not choose the frozen-backbone model because it was less accurate, even though it trained faster.

7. Conclusion
-The experiments show that data augmentation can improve leaf recognition. Changing colours was especially useful, while geometric changes alone did not help much. Combining both gave the best results.
-However, the model still has problems recognising completely new photos. More testing with real phone camera images is needed before choosing the final version.
