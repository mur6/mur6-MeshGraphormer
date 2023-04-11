import torch
from torch import nn




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x, flags):
        results = []
        for flag in flags:
            if flag:
                ans = x * x
            else:
                ans = torch.sqrt(x)
            results.append(ans)
        return torch.tensor(results)

model= MyModel()

x = torch.tensor(5.0)
a1 = torch.tensor([True, False])
print(x, a1)
r = model(x, a1)
print(r)
torch.onnx.export(
    model, (x, a1), "model.onnx", opset_version=13)
#    example_outputs=(traced_model_true(x, True), traced_model_false(x, False))



# a2 = torch.tensor(False)
# print(bool(a1), bool(a2))
# traced_model_true = torch.jit.trace(model, (x, a1))
# traced_model_false = torch.jit.trace(model, (x, a2))
# scripted_model = torch.jit.script(model)
# print(traced_model_true)

# @torch.jit.script
# def scripted_model(x, flag):
#     if flag:
#         # flag=Trueの処理
#     else:
#         # flag=Falseの処理

