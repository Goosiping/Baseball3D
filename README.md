# Baseball3D
This project aims to predict the 3D position of the baseball in the given videos. We use yolo v8 to detect the 2D bounding boxes of the baseball from multiple videos, then we use the camera parameters to calculate the 3D position of the baseball.

## Environment setups
```bash
pip install ultralytics
pip install opencv-python
```