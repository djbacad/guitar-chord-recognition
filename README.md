Real-time Guitar Chord Recognition
==================================

![prediction](https://github.com/djbacad/guitar-chord-recognition/assets/61301478/c916eee2-7752-486b-b8d4-5d5839b30772)

This is a prototype project aimed to assess the feasibility of employing computer vision techniques to identify guitar chords in real-time.

Notes:
- Using only 208 images a model was trained to identify three chords (classes) namely CMaj, DMaj, and GMaj.
- To address data scarcity, data augmentation techniques were employed. The necesasry keras custom objects were utilized for robustness.
- Using the Tensorflow framework, the final model was created using TL&FT technique based on the EfficientNetV2B0 architecture and achieved an accuracy of 89.9% on the test set.

Hardware: 
- Nvidia GeForce RTX 2060 Mobile GPU
- Ryzen 7 4800H CPU

Development Environment:
- WSL

Limitations:
- The model is only able to predict 3 classes (C, D, G)
- The results are limited to the data collected.

Future Work:
- Include audio data in the prediction



