import onnx
from onnx import TensorProto

onnx_path = "best_n_300.onnx"

model = onnx.load(onnx_path)

print("========== Basic ==========")
print("ir_version:", model.ir_version)
print("producer_name:", model.producer_name)
print("producer_version:", model.producer_version)

print("\n========== Opset ==========")
for opset in model.opset_import:
    domain = opset.domain if opset.domain else "ai.onnx"
    print(domain, opset.version)

print("\n========== Metadata ==========")
meta = {p.key: p.value for p in model.metadata_props}
for k, v in meta.items():
    print(f"{k}: {v}")

def value_info_shape(v):
    t = v.type.tensor_type
    elem_type = TensorProto.DataType.Name(t.elem_type)
    dims = []
    for d in t.shape.dim:
        if d.dim_param:
            dims.append(d.dim_param)   # dynamic axis
        elif d.dim_value:
            dims.append(d.dim_value)   # static axis
        else:
            dims.append("?")
    return elem_type, dims

print("\n========== Inputs ==========")
for inp in model.graph.input:
    dtype, shape = value_info_shape(inp)
    print(inp.name, dtype, shape)

print("\n========== Outputs ==========")
for out in model.graph.output:
    dtype, shape = value_info_shape(out)
    print(out.name, dtype, shape)

print("\n========== Node types ==========")
node_types = {}
for node in model.graph.node:
    node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

for k, v in sorted(node_types.items()):
    print(k, v)