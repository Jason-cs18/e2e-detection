# clear all intermediate checkpoints for commit
import os
import glob

def main():
    current_path = os.getcwd()
    torch_models = glob.glob(current_path + '/**/*.pth', recursive=True)
    onnx_models = glob.glob(current_path + '/**/*.onnx', recursive=True)
    tensorrt_models = glob.glob(current_path + '/**/*.engine', recursive=True)
    # print(current_path)
    # delete *.pth, *.onnx, *.engine
    for model in torch_models:
        os.remove(model)
    for model in onnx_models:
        os.remove(model)
    for model in tensorrt_models:
        os.remove(model)
    # pass


if __name__ == '__main__':
    main()