import onnxruntime as ort

print("Cargando modelo...")
session = ort.InferenceSession("aves_resnet.onnx")

print("\n=== ENTRADAS ===")
for inp in session.get_inputs():
    print("Nombre:", inp.name)
    print("Forma:", inp.shape)
    print("Tipo:", inp.type)
    print("-" * 30)

print("\n=== SALIDAS ===")
for out in session.get_outputs():
    print("Nombre:", out.name)
    print("Forma:", out.shape)
    print("Tipo:", out.type)
    print("-" * 30)
