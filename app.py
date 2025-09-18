from ultralytics import YOLO
model = YOLO("best.pt")


#results = model(source=1, show=True, save=True)
results = model("test_digit.png", show=True, save=True)

for r in results:
    br = r.boxes
    cl = r.names
    lt = r.__len__()
    tj = r.to_json()

    print(tj)