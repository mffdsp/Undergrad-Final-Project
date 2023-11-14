# A Transformer- based  Architecture Neural Network Approach to Email Message Autocomplete 

This research project aims to develop and explore an neural network architecture based on transformers, with the primary objective of improving the automatic completion of email messages. The study encompasses a detailed process of creating, training, and evaluating the architecture while investigating the influence of various hyperparameters and layers. The central emphasis is on designing a transformer model capable of effectively capturing complex long-range dependencies inherent in email communications.

| **Hyperparams**         | **Search Space**        | **Selected Value** | **Base-line (GPT-2 )** |
|-------------------------|-------------------------|--------------------|------------------------|
| Learning Rate           | [0.0001, 0.00001]       | 0.00001            | 0.00001                |
| Block Size              | [128, 1024]             | 384                | 1024                   |
| Batch Size              | [32, 512]               | 56                 | 512                    |
| Number of Epochs        | [6000, 8000]            | 8000               | -                      |
| Dropout Rate            | [0.1, 0.2]              | 0.1                | 0.1                    |
| Number of Layers/Heads  | [4, 12]                 | 8                  | 12                     |
| Embedding dimension     | [156, 768]              | 384                | 768                    |
| Weight Decay            | [0.01, 0.000001]        | 0.000001           | 0.01                   |
| Optimizer               | 'Adam', 'AdamW'         | AdamW              | -                      |
| Activation function     | -                       | SiLU               | GeluNew                |


## Methodology

### Data Cleansing
Before training the model, an extensive data cleansing procedure is executed to ensure the quality and relevance of the input data.

### Preliminary Training
Due to hardware resource constraints, the architecture undergoes initial training on a substantial corpus of publicly available web texts. This provides a foundation for the subsequent stages of the research.

### Evaluation
The model is rigorously evaluated on an independent test dataset to assess its performance and generalization capabilities.

### Fine-tuning
Subsequent fine-tuning is performed using individualized user email data. This stage is crucial for adapting the model to specific user patterns and preferences.

### Hyperparameter Analysis
A thorough analysis of the impact of diverse hyperparameters on the model's performance is conducted. This includes a comparative assessment of performance metrics both before and after the fine-tuning process.

## Objectives

1. **Architecture Construction**: To study the creation of the transformer-based architecture in the context of email autocompletion mechanisms.

2. **Resource Optimization**: Explore the applicability of transformer models in email message generation, considering hardware constraints and resource utilization.

3. **Performance Metrics**: Identify and analyze performance metrics to understand the strengths and limitations of the developed model.

4. **Parameter Tuning**: Investigate the effects of hyperparameters on the model's performance and fine-tune accordingly for optimal results.

## Results and Conclusion

Through these objectives, this study aims to provide insights into the construction and application of transformer models in the realm of email communication. The findings will contribute to understanding limitations, suggesting areas of improvement, guiding parameter tuning, and establishing a solid foundation for implementing this technology at the code level in email communication applications.