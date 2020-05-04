import matplotlib.pyplot as plt

from functions import Functions

from skimage import data

class FunctionViewer(object):
    def __init__(self, function_list):
        self.function_list = function_list

    def run(self, img0, img1, p0, p1, p2, p3, p4):
        fig, axs = plt.subplots(len(self.function_list), 3, figsize=(15, len(self.function_list)*3))
        for i, function in enumerate(self.function_list):
            img_out = Functions.execute(function, img0, img1, p0, p1, p2, p3, p4)
            axs[i, 0].imshow(img0, cmap=plt.get_cmap('gray'))
            axs[i, 1].imshow(img1, cmap=plt.get_cmap('gray'))
            axs[i, 2].imshow(img_out, cmap=plt.get_cmap('gray'))
            axs[i, 1].set_title('Function ' + str(function) + ', p0=' + str(p0)+ ', p1=' + str(p1)+ ', p2=' + str(p2)+ ', p0=' + str(p2)+ ', p3=' + str(p3))
        plt.axis('off')
        plt.savefig('test.png', dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    src_img = data.astronaut()
    img0 = src_img[:, :, 0]
    img1 = src_img[:, :, 1]
    FunctionViewer(range(4, 51)).run(img0, img1, 64, 0, 0, 0, 0)