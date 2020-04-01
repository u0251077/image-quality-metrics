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

  ```bash
  $ cd ./src
  $ python metric.py --gt ../image/org.png --target ../image/shift.png
  >>> Using TensorFlow backend.
  >>> Mean Square Error: 0.031699415
  >>> PSNR: 14.989488124847412 (dB)
  >>> SSIM: 0.5323553827937944

  $ python metric.py --gt ../image/org.png --target ../image/blur.png
  >>> Using TensorFlow backend.
  >>> Mean Square Error: 0.0009565791
  >>> PSNR: 30.192790031433105 (dB)
  >>> SSIM: 0.9131808242642852
  ```

  ​

