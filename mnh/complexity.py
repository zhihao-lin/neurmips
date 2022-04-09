import torch
from ptflops import get_model_complexity_info

class InputData():
    def __init__(self, data):
        self.data = data
    
    def __call__(self, inputs):
        return self.data

def calculate_complexity(net, input_data):
    data_constructer = InputData(input_data)
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(net, (0, 0), as_strings=True,
                                           input_constructor=data_constructer,
                                           print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))