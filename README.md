## ALERT open dataset 👋

1. ALERT dataset download: https://doi.org/10.6084/m9.figshare.28244525.v2
  - We strongly recommend the linux environment.
  - Using tar -xvzf ALERT_train.tar.gz for decompression
  - Locate the ALERT_train directory on the same level of ALERT files
2. Make ALERT dataset to pickle files: using ALERT_makeDataset.py
  - We can manipulate the samples (e.g., observation window control, infromation cropping, ...) in this file. 
  - Usage example: python3 ALERT_makeDataset.py common/extend cropO/cropX/CA sample_size,   ex) python3 ALERT_makeDataset.py common cropO 500
3. Setting environment: using ALERT_setting.py
4. Model configurations: using ALERT_models.py
  - We can modify the details of models in this file.
5. Validation of benchmarking: using ALERT_main,py
  - We can conduct extensive experiments (e.g., few-shot adaptation, beta learning, ...) in this file.
