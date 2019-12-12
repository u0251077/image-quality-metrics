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
  $ pip install virtualenv venv
  $ source ./venv/bin/activate
  # install requirement
  $ pip install -r requirements.txt
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

