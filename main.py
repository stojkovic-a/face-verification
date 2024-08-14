import train2
import faceNet
import os

CUDA_LAUNCH_BLOCKING = 1
if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    model_path = './Models/m4.pt'
    train2.run(faceNet.FaceNet, model_path=model_path)
    # train2.load_test_model(faceNet.FaceNet, 95897, 4959)
    # 30 (40) 50 i 80
