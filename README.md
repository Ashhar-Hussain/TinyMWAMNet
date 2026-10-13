# TinyMWAMNet

Code for the article **"Lossless In-Domain, Lossy under Shift: Post-Training Quantization Can Make Tiny Machine Learning Wildfire Detectors Miss Fires"** by Ashhar Hussain (Department of Computer Science and Information Technology, University of Kotli, Pakistan), submitted to *Engineering Applications of Artificial Intelligence* (Elsevier).

The study distils the MWAMNet wildfire-detection ensemble into microcontroller-sized students (TinyMWAMNet), converts them to 8-bit integers (INT8) by post-training quantization (PTQ) and quantization-aware training (QAT), and evaluates every model in-domain and on two shifted datasets.

## Notebooks

Every experiment runs on free Google Colab. Outputs are written to Google Drive (`MyDrive/TinyMWAMNet/`). If a session disconnects, choose **Runtime > Run all** again: finished steps are skipped.

| Order | Notebook | Runtime | Approx. time | Content |
|---|---|---|---|---|
| 1 | `01_Data_and_Teacher.ipynb` | T4 GPU | 45–60 min | datasets, leakage audit, MWAMNet teacher (5 seeds), soft labels, weight-only bit-width sweep |
| 2 | `02_Teacher_Quantization_CPU.ipynb` | CPU | 1–6 h | FP16, dynamic-range and full-integer INT8 teachers; latency, TFLM memory, Vela estimate |
| 3 | `03_TinyMWAMNet_Students.ipynb` | T4 GPU | 3–4 h | students under four training regimes, size family, PTQ and QAT, deployment metrics |
| 4 | `04_QAT_fix_and_ReLU6.ipynb` | T4 GPU | 2.5–3 h | QAT with range warm-up, INT8 scoring with LiteRT built-in kernels, ReLU6 students |
| 5 | `05_Analysis_Overflow_XAI.ipynb` | CPU | 45–60 min | runtime agreement, activation overflow, statistics, occlusion maps |
| 6 | `07_Error_Direction_and_TFLM.ipynb` | CPU | about 1 h | logit shift and decision margins; key models scored with TensorFlow Lite Micro |
| 7 | `06_Figures_and_Tables.ipynb` | CPU | about 5 min | all figures and LaTeX tables of the article |

Software: TensorFlow 2.21 with `tf_keras`, TensorFlow Model Optimization Toolkit 0.8.1, LiteRT, TensorFlow Lite Micro Python runtime (`tflite-micro`, build of 1 December 2025) and Vela 5.2.0. Each notebook installs what it needs.

## Data

The notebooks download the three public datasets from Kaggle. Add your Kaggle credentials as Colab secrets named `KAGGLE_USERNAME` and `KAGGLE_KEY`.

| Role | Dataset | Kaggle identifier | Reference |
|---|---|---|---|
| In-domain | Forest Fire Dataset | `alik05/forest-fire-dataset` | Khan et al., *Mob. Inf. Syst.* 2022, https://doi.org/10.1155/2022/5358359 |
| Shift-1 | The Wildfire Dataset | `elmadafri/the-wildfire-dataset` | El-Madafri et al., *Forests* 2023, https://doi.org/10.3390/f14091697 |
| Shift-2 | D-Fire (test split) | `sayedgamal99/smoke-fire-detection-yolo` | de Venâncio et al., *Neural Comput. Appl.* 2022, https://doi.org/10.1007/s00521-022-07467-z |

## Supplementary material

Supplementary Tables S1–S5 of the article (S1 QAT without the range warm-up; S2 the size family; S3 activation overflow; S4 runtime agreement; S5 key models scored with TFLM): https://github.com/Ashhar-Hussain/supplementary

## Contact

Ashhar Hussain (author), Department of Computer Science and Information Technology, Faculty of Computing and Engineering, University of Kotli, Kotli 11100, Azad Jammu and Kashmir, Pakistan (asharhussain.csit@uokajk.edu.pk)
