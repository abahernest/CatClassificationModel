from model import *

my_image = "my_image.jpg"  

# Preprocess the image to fit your algorithm.
fullname = "images/" + my_image
image = np.array(imageio.imread(fullname))
image = image/255.
my_image = skimage.transform.resize(image, (num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = predict(model_instance["w"], model_instance["b"], my_image)


print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
plt.imshow(image)