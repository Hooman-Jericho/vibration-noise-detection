# Attribution

This project is a re-engineering of
[MiladSoleymani/vibration_noise_detection](https://github.com/MiladSoleymani/vibration_noise_detection)
(MIT). The original problem statement, data description and the overall idea
(Butterworth high-pass → PSD → SVM / XGBoost / KNN / decision tree → majority voting) are
the original author's work.

Re-implemented here: package structure, data discovery, signal processing defaults,
feature extraction, model registry, nested group-wise evaluation, persistence, inference, CLI,
tests and CI. No original source file is carried over unchanged. If you publish this repository,
keep this notice and state clearly which parts are yours.
