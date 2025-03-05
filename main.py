import os
from model import Model
from dataset import Dataset
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import ezkl
import warnings
warnings.filterwarnings('ignore')
ezkl.gen_settings("C:\\Users\\Richa\\PycharmProjects\\CECS-574---ZKML\\model.onnx")


if __name__ == '__main__':
    # Setting up dataset and model
    dataset = Dataset()

    x_train, x_test, y_train, y_test =  dataset.split_dataset()

    model = Model()

    file_path = 'C:\\Users\\Richa\\PycharmProjects\\CECS-574---ZKML\model.pkl'

    if not os.path.exists(file_path):
        print('Saving New Model')
        model.train(x_train, y_train)
        model.test(x_test, y_test)
        model.save_model()
    else:
        print('Loading Model')
        model.load_model()
        model.test(x_test, y_test)

    # Converting to ONNX model
    input_type = FloatTensorType([None, model.get_model().n_features_in_])

    # onnx_model = skl2onnx.convert_sklearn(model.get_model(), initial_types=[("input", input_type)])
    onnx_model = skl2onnx.convert_sklearn(
        model.get_model(),
        initial_types=[("input", input_type)],
        target_opset=12  # Lower opset may improve compatibility
    )

    # Save ONNX model
    with open("C:\\Users\\Richa\\PycharmProjects\\CECS-574---ZKML\\model.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("Model successfully converted to ONNX format.")
