import os
import torch.onnx
from model import Model
from dataset import Dataset
from LogisticRegression import LogisticRegressionTorch
from skl2onnx.common.data_types import FloatTensorType
import ezkl
import asyncio
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    # Setting up dataset and model
    dataset = Dataset()
    x_train, x_test, y_train, y_test = dataset.split_dataset()

    model = Model()

    file_path = 'model.pkl'
    if not os.path.exists(file_path):
        print('Saving New Model')
        model.train(x_train, y_train)
        model.test(x_test, y_test)
        model.save_model()
    else:
        print('Loading Model')
        model.load_model()

    onnx_path = 'model.onnx'
    if not os.path.exists(onnx_path):
        # Converting to ONNX model
        input_type = FloatTensorType([None, x_train.shape[1]])

        onnx_model = LogisticRegressionTorch(model.get_weight(), model.get_bias())
        dummy_input = torch.randn(1, len(model.get_weight()))
        torch.onnx.export(onnx_model, dummy_input, onnx_path, export_params=True, input_names=['input'], output_names=['output'])

        print("Model successfully converted to ONNX format.")

    ezkl.gen_settings('model.onnx')
    ezkl.compile_circuit("model.onnx", "model.ezkl", "settings.json")
    loop = asyncio.get_event_loop()
    # Run get_srs
    loop.run_until_complete(ezkl.get_srs(settings_path='settings.json', srs_path='kzg.srs'))
    ezkl.setup("model.ezkl", "vk.key", "pk.key", "kzg.srs")



