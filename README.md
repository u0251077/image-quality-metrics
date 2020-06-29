# image quality metrics

![](https://i.imgur.com/dX8hDkq.png)

## start

- clone 

  ```bash
  $ git clone https://github.com/u0251077/image-quality-metrics.git
  ```

- install requirement

  ```bash
  # use virtualenv
  $ virtualenv venv
  $ source ./venv/bin/activate
  # install requirement
  $ pip install keras==2.3.1
  $ pip install tensorflow==2.1.0
  $ pip install opencv-python==4.2.0.32
  $ pip install scikit-image==0.14.5
  ```

- metrics image & output

  + modify code
  ```python=
  gt = "./image-quality-metrics/output_org20/1.jpg"
  predict = "./image-quality-metrics/output_blur20/1.jpg"

  M = Metric()
  result = M.calculateCosSim(gt,predict)
  print("cos_sim:", result)
  result = M.calculateMSE(gt,predict)
  print("MSE:", result)
  result = M.calculatePSNR(gt,predict)
  print("PSNR:", result)
  result = M.calculateSSIM(gt,predict)
  print("SSIM:", result)
  ```
  + bash
  ```bash=
  $ cd ./src
  $ python metric.py 
  >>> Using TensorFlow backend.
  >>> ('cos_sim:', 0.28668737)
  >>> ('MSE:', 0.019655704)
  >>> ('PSNR:', 17.06511378288269)
  >>> ('SSIM:', 0.2214987835228802)
  ```

  ​

