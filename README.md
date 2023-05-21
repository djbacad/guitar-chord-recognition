Real-time Guitar Chord Recognition
==================================

![prediction](https://github.com/djbacad/guitar-chord-recognition/assets/61301478/42779f15-ca82-4fdd-ae8f-3a0f98a1d91f)

This is a prototype project aimed to assess the feasibility of employing computer vision techniques to identify guitar chords in real-time.

### Project Highlights:
- Using only 208 images a model was trained to identify three chords (classes) namely CMaj, DMaj, and GMaj.
- To address data scarcity, data augmentation techniques were employed. The necesasry keras custom objects were utilized for robustness.
- Using the [Tensorflow](https://www.tensorflow.org/) framework, the final model was created using TL&FT technique based on the EfficientNetV2B0 architecture and achieved an accuracy of 89.9% on the test set.

### Hardware:
- Nvidia GeForce RTX 2060 Mobile GPU
- Ryzen 7 4800H CPU

### Operating System:
- WSL

### Limitations:
- The model is only able to predict 3 classes (C, D, G)
- The results are limited to the data collected.

### Future Work:
- Include audio data in the prediction
- Collect more images and include the entire neck/fretboard of the guitar as vision input

### Try the code:

Place your video file inside test folder, navigate inside the src/models folder and issue the following command:
```cmd
python predict.py <your_video_filename>
```

### Credits/Citations:

Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., Corrado, G. S., Davis, A., Dean, J., Devin, M., Ghemawat, S., Goodfellow, I., Harp, A., Irving, G., Isard, M., Jozefowicz, R., Jia, Y., Kaiser, L., Kudlur, M., Levenberg, J., Mané, D., Schuster, M., Monga, R., Moore, S., Murray, D., Olah, C., Shlens, J., Steiner, B., Sutskever, I., Talwar, K., Tucker, P., Vanhoucke, V., Vasudevan, V., Viégas, F., Vinyals, O., Warden, P., Wattenberg, M., Wicke, M., Yu, Y., & Zheng, X. (2015). TensorFlow, Large-scale machine learning on heterogeneous systems [Computer software]. https://doi.org/10.5281/zenodo.4724125

Tan, M. &amp; Le, Q.. (2021). EfficientNetV2: Smaller Models and Faster Training. <i>Proceedings of the 38th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 139:10096-10106 Available from https://proceedings.mlr.press/v139/tan21a.html.

