# DermiScan

## Abstract

DermiScan is a deep learning-based application developed to provide a reliable and convenient method for facial acne diagnosis. With the increasing demand for dermatological services and the shortage of physicians, our aim is to leverage advanced computer vision and deep learning technologies to assist users in obtaining accurate acne assessments. Utilizing a diverse dataset gathered from public dermatological databases, our model is trained using PyTorch and incorporates OpenCV for enhanced image processing. The primary goal is to deliver dermatologist-level acne detection and classification, validated against a separate dataset annotated by dermatology professionals.

## Introduction

The application of artificial intelligence and machine learning in healthcare has opened new avenues for assisting healthcare providers in diagnosing and flagging potential conditions. According to the American Academy of Dermatology Association, approximately 50 million Americans suffer from acne, making it a significant concern that impacts both physical and psychological well-being. Traditional dermatological consultations can be expensive, time-consuming, and may involve trial-and-error treatments. Our project, DermiScan, aims to address these issues by providing an accessible and efficient solution that leverages advancements in computer vision and deep learning to facilitate acne diagnosis.

## Background

Acne affects individuals across various age groups, causing psychological distress and impacting their quality of life. The conventional diagnosis process involves visiting a dermatologist, running various tests, and receiving prescriptions. However, this approach has limitations in terms of accessibility and cost. The project utilizes computer vision and deep learning technologies, specifically convolutional neural networks (CNN), to detect and classify acne types from images. By exploring alternatives to existing solutions like the ResNet 152 model, it aims to offer a more inclusive and effective solution.

## Methodology

### Data Collection

- **Datasets**: Public dermatological databases, including Derment.com.
- **Data Components**: Images and annotations labeling acne positions for better model training.

### Model Selection

- **Deep Learning Framework**: PyTorch.
- **Model Architecture**: Faster R-CNN with a ResNet 50 backbone, selected for its balance between accuracy and speed.

### Training Process

1. **Data Transformation**: Images undergo resizing, tensor conversion, and normalization.
2. **Model Initialization**: Using pre-trained weights for the Faster R-CNN model.
3. **Loss Computation**: Comparing predictions with ground truth annotations.
4. **Metrics Calculation**: Precision, recall, F1 score, and Mean Average Precision (mAP) to evaluate model performance.
5. **Epochs**: Initially set to 10, increased to 17 for better performance.

### Challenges

- Ensuring model accuracy and diversity.
- Addressing privacy concerns associated with facial data.
- Training on complex datasets for accurate acne classification.

## Results

The model showed promising results in detecting acne, with consistent improvements across training epochs:

- **Training Loss**: Decreased from 1.2682 to 1.0383, indicating effective learning.
- **Validation Metrics**: Precision improved, showing the model's ability to detect acne.
- **F1 Score**: Increased from 10.68% to 14.69%, indicating balanced training.
- **mAP**: Showed an upward trend, suggesting improvements in detecting relevant instances.

### Observations

- The model faced challenges with complex datasets, affecting precision and mAP.
- Future improvements include training on less complex datasets and applying advanced regularization techniques to avoid overfitting.

## Conclusion

DermiScan offers a valuable tool for acne detection and classification, potentially saving time, effort, and cost for patients. While the model shows promise, further improvements are needed to enhance precision and classification accuracy. The project has the potential to assist dermatologists in clinical settings, fostering trust between patients and healthcare providers. Future work includes refining the model and seeking expert feedback for further development and treatment recommendations.

## Future Work

- **Dataset Refinement**: Training on less complex datasets to improve model performance.
- **Regularization Techniques**: Implementing advanced methods to avoid overfitting.
- **Expert Feedback**: Engaging dermatologists for feedback and integrating treatment recommendations.
- **Privacy and Security**: Ensuring user data is handled securely and respectfully.

**Note: This project is still ongoing, and we am still actively working on improvements and new features to enhance its effectiveness and usability. Any contributions and feedback from the community are welcome.**

## Contributors
- **[Contributor 1](https://github.com/moe-elh)**
- **[Contributor 2](https://github.com/NaiaAlmoudareys)**

## References

- Lewis, et al. (2022). [Study on Dermatology Specialty](#)
- Anon, et al. (2016). [AI and Machine Learning in Healthcare](#)
- Morshed, et al. (2023). [Impact of Acne Vulgaris on Psychological Health](#)
- Cleveland Clinic. (2020). [Acne Information](#)
- Shehmir Javaid. (2020). [Computer Vision in Healthcare](#)
- IBM. (2023). [Convolutional Neural Networks](#)
- Adrian RoseBrock. (2021). [Comparison of PyTorch Models](#)
- Anon. (2020). [ResNet Model Analysis](#)
- Arya Ahmed. (n.a.). [Precision and Recall in Machine Learning](#)
- Buhl. (2023). [F1 Score Explanation](#)
- Team, et al. (n.d.). [Validation Recall Metric](#)
- YOHANANDAN, S. (2020). [Mean Average Precision in Object Detection](#)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We would like to thank the contributors and collaborators who made this project possible, including the developers of PyTorch and OpenCV. Special thanks to the dermatology professionals who provided valuable insights and feedback.
