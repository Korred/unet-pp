from model.unetpp import UNetPlusPlus

# Build the model and print the summary
input_shape = (256, 256, 3)
num_classes = 1
model = UNetPlusPlus(input_shape, num_classes).model
model.summary()
